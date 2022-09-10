import networkx as nx
import torch
from sacred.run import Run
from logging import Logger
import numpy as np

from data_node2vec import CATEGORY_IDS
import models
import utils



@torch.no_grad()
def eval_link_prediction(model, triples_loader, text_dataset, entities, relations, 
                         epoch, emb_batch_size, _run: Run, _log: Logger,
                         prefix='', max_num_batches=None,
                         filtering_graph=None, new_entities=None,
                         return_embeddings=False, device='cpu'):
    compute_filtered = filtering_graph is not None
    mrr_by_position = torch.zeros(3, dtype=torch.float).to(device)
    mrr_pos_counts = torch.zeros_like(mrr_by_position)

    rel_categories = triples_loader.dataset.rel_categories.to(device)
    mrr_by_category = torch.zeros([2, 4], dtype=torch.float).to(device)
    mrr_cat_count = torch.zeros([1, 4], dtype=torch.float).to(device)

    hit_positions = [1, 3, 10]
    k_values = torch.tensor([hit_positions], device=device)
    hits_at_k = {pos: 0.0 for pos in hit_positions}
    mrr = 0.0
    mrr_filt = 0.0
    hits_at_k_filt = {pos: 0.0 for pos in hit_positions}

    if device != torch.device('cpu'):
        model = model.module

    if isinstance(model, models.inductive_lp.InductiveLinkPrediction):
        print('in')
        num_entities = entities.shape[0]
        num_relations = relations.shape[0]
        if compute_filtered:
            max_ent_id = max(filtering_graph.nodes)
        else:
            max_ent_id = entities.max()

        max_rel_id = entities.max()

        ent2idx = utils.make_ent2idx(entities, max_ent_id)
        rel2idx = utils.make_rel2idx(relations, max_rel_id)
    else:
        # Transductive models have a lookup table of embeddings
        num_entities = model.ent_emb.num_embeddings
        num_relations = model.rel_emb.num_embeddings
        ent2idx = torch.arange(num_entities)
        entities = ent2idx

    ### Create embedding lookup table for evaluation -- for entities
    ent_emb = torch.zeros((num_entities, model.dim), dtype=torch.float,
                              device=device)

    idx = 0
    num_iters = np.ceil(num_entities / emb_batch_size)
    iters_count = 0
    while idx < num_entities:
        # Get a batch of entity IDs and encode them
        batch_ents = entities[idx:idx + emb_batch_size]

        if isinstance(model, models.inductive_lp.InductiveLinkPrediction):
            # Encode with entity descriptions
            data = text_dataset.get_entity_description(batch_ents)
            text_tok_ent, text_mask_ent, text_len = data
            batch_emb = model(text_tok_ent.unsqueeze(1).to(device),
                              text_mask_ent.unsqueeze(1).to(device))

        else:
            # Encode from lookup table
            batch_emb = model(batch_ents)

        ent_emb[idx:idx + batch_ents.shape[0]] = batch_emb

        iters_count += 1
        if iters_count % np.ceil(0.2 * num_iters) == 0:
            _log.info(f'[{idx + batch_ents.shape[0]:,}/{num_entities:,}]')

        idx += emb_batch_size

    ent_emb = ent_emb.unsqueeze(0)

    ### Create embedding lookup table for evaluation -- for relations
    rel_emb = torch.zeros((num_relations, model.dim), dtype=torch.float,
                              device=device)

    idx = 0
    num_iters = np.ceil(num_relations / emb_batch_size)
    iters_count = 0
    while idx < num_relations:
        # Get a batch of relation IDs and encode them
        batch_rels = relations[idx:idx + emb_batch_size]

        if isinstance(model, models.BertCombinedLP):
            # Encode with entity descriptions
            data = text_dataset.get_relation_description(batch_rels)
            text_tok_rel, text_mask_rel, text_len = data

            batch_features = text_dataset.get_relation_features(batch_rels)

            batch_emb = model(text_tok_rel=text_tok_rel.unsqueeze(1).to(device),
                              text_mask_rel=text_mask_rel.unsqueeze(1).to(device),
                              relation_features=batch_features.to(device))

        elif isinstance(model, models.BertNode2VecSame) or isinstance(model, models.BertNode2VecDiff) or isinstance(model, models.BOWNode2Vec) or isinstance(model, models.DKRLNode2Vec):
            batch_features = text_dataset.get_relation_features(batch_rels)
            batch_emb=model(relation_features=batch_features.to(device))


        elif isinstance(model, models.inductive_lp.InductiveLinkPrediction):
            # Encode with entity descriptions
            data = text_dataset.get_relation_description(batch_rels)
            text_tok_rel, text_mask_rel, text_len = data
            batch_emb = model(text_tok_rel=text_tok_rel.unsqueeze(1).to(device),
                              text_mask_rel=text_mask_rel.unsqueeze(1).to(device))

        else:
            # Encode from lookup table
            batch_emb = model(batch_rels)

        rel_emb[idx:idx + batch_rels.shape[0]] = batch_emb

        iters_count += 1
        if iters_count % np.ceil(0.2 * num_iters) == 0:
            _log.info(f'[{idx + batch_rels.shape[0]:,}/{num_relations:,}]')

        idx += emb_batch_size

    rel_emb = rel_emb.unsqueeze(0)

    num_predictions = 0
    _log.info('Computing metrics on set of triples')
    total = len(triples_loader) if max_num_batches is None else max_num_batches
    for i, triples in enumerate(triples_loader):
        if max_num_batches is not None and i == max_num_batches:
            break

        heads, tails, rels = torch.chunk(triples, chunks=3, dim=1)
        # Map entity IDs to positions in ent_emb
        heads = ent2idx[heads].to(device)
        tails = ent2idx[tails].to(device)
        rels = rel2idx[rels].to(device)

        assert heads.min() >= 0
        assert tails.min() >= 0
        assert rels.min() >= 0

        # Embed triple
        head_embs = ent_emb.squeeze()[heads]
        tail_embs = ent_emb.squeeze()[tails]
        rel_embs = rel_emb.squeeze()[rels]
        #rel_embs = model.rel_emb(rels.to(device))

        # Score all possible heads and tails
        heads_predictions = model.score_fn(ent_emb, tail_embs, rel_embs)
        tails_predictions = model.score_fn(head_embs, ent_emb, rel_embs)

        pred_ents = torch.cat((heads_predictions, tails_predictions))
        true_ents = torch.cat((heads, tails))

        num_predictions += pred_ents.shape[0]
        reciprocals, hits = utils.get_metrics(pred_ents, true_ents, k_values)
        mrr += reciprocals.sum().item()
        hits_sum = hits.sum(dim=0)
        for j, k in enumerate(hit_positions):
            hits_at_k[k] += hits_sum[j].item()

        if compute_filtered:
            filters = utils.get_triple_filters(triples, filtering_graph,
                                               num_entities, ent2idx)
            heads_filter, tails_filter = filters
            # Filter entities by assigning them the lowest score in the batch
            filter_mask = torch.cat((heads_filter, tails_filter)).to(device)
            pred_ents[filter_mask] = pred_ents.min() - 1.0

            reciprocals, hits = utils.get_metrics(pred_ents, true_ents, k_values)
            mrr_filt += reciprocals.sum().item()
            hits_sum = hits.sum(dim=0)
            for j, k in enumerate(hit_positions):
                hits_at_k_filt[k] += hits_sum[j].item()

            reciprocals = reciprocals.squeeze()
            if new_entities is not None:
                by_position = utils.split_by_new_position(triples,
                                                          reciprocals,
                                                          new_entities)
                batch_mrr_by_position, batch_mrr_pos_counts = by_position
                mrr_by_position += batch_mrr_by_position
                mrr_pos_counts += batch_mrr_pos_counts

            if triples_loader.dataset.has_rel_categories:
                by_category = utils.split_by_category(triples,
                                                      reciprocals,
                                                      rel_categories)
                batch_mrr_by_cat, batch_mrr_cat_count = by_category
                mrr_by_category += batch_mrr_by_cat
                mrr_cat_count += batch_mrr_cat_count

        if (i + 1) % int(0.5 * total) == 0:
            _log.info(f'[{i + 1:,}/{total:,}]')

    _log.info(f'The total number of predictions is {num_predictions:,}')
    for hits_dict in (hits_at_k, hits_at_k_filt):
        for k in hits_dict:
            hits_dict[k] /= num_predictions

    mrr = mrr / num_predictions
    mrr_filt = mrr_filt / num_predictions

    log_str = f'{prefix} mrr: {mrr:.4f}  '
    _run.log_scalar(f'{prefix}_mrr', mrr, epoch)
    for k, value in hits_at_k.items():
        log_str += f'hits@{k}: {value:.4f}  '
        _run.log_scalar(f'{prefix}_hits@{k}', value, epoch)

    if compute_filtered:
        log_str += f'mrr_filt: {mrr_filt:.4f}  '
        _run.log_scalar(f'{prefix}_mrr_filt', mrr_filt, epoch)
        for k, value in hits_at_k_filt.items():
            log_str += f'hits@{k}_filt: {value:.4f}  '
            _run.log_scalar(f'{prefix}_hits@{k}_filt', value, epoch)

    _log.info(log_str)

    if new_entities is not None and compute_filtered:
        mrr_pos_counts[mrr_pos_counts < 1.0] = 1.0
        mrr_by_position = mrr_by_position / mrr_pos_counts
        log_str = ''
        for i, t in enumerate((f'{prefix}_mrr_filt_both_new',
                               f'{prefix}_mrr_filt_head_new',
                               f'{prefix}_mrr_filt_tail_new')):
            value = mrr_by_position[i].item()
            log_str += f'{t}: {value:.4f}  '
            _run.log_scalar(t, value, epoch)
        _log.info(log_str)

    if compute_filtered and triples_loader.dataset.has_rel_categories:
        mrr_cat_count[mrr_cat_count < 1.0] = 1.0
        mrr_by_category = mrr_by_category / mrr_cat_count

        for i, case in enumerate(['pred_head', 'pred_tail']):
            log_str = f'{case} '
            for cat, cat_id in CATEGORY_IDS.items():
                log_str += f'{cat}_mrr: {mrr_by_category[i, cat_id]:.4f}  '
            _log.info(log_str)

    if return_embeddings:
        out = (mrr, ent_emb, rel_emb)
    else:
        out = (mrr, None)

    return out
