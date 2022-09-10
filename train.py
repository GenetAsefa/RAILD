import os
import os.path as osp
import networkx as nx
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sacred.run import Run
from logging import Logger
from sacred import Experiment
from sacred.observers import MongoObserver
from transformers import BertTokenizer, DistilBertTokenizer, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np

from data import GraphDataset, TextGraphDataset, GloVeTokenizer
import utils
from evaluation import eval_link_prediction


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ex = Experiment()
ex.logger = utils.get_logger()
# Set up database logs
uri = os.environ.get('DB_URI')
database = os.environ.get('DB_NAME')
if all([uri, database]):
    ex.observers.append(MongoObserver(uri, database))


@ex.config
def config():
    dataset = 'umls'
    dataset_setting = 's-inductive'
    dim = 128
    model = 'raild'
    rel_model = 'transe'
    loss_fn = 'margin'
    encoder_name = None
    regularizer = 0
    max_len = 32
    num_negatives = 64
    lr = 2e-5
    use_scheduler = True
    batch_size = 64
    emb_batch_size = 512
    eval_batch_size = 64
    max_epochs = 40
    checkpoint = None
    use_cached_ent_text = False
    use_cached_rel_text = False
    output_directory = 'output/'

#@ex.capture
#@torch.no_grad()
eval_link_prediction = ex.capture(eval_link_prediction)

@ex.command
def link_prediction(dataset, dataset_setting, dim, model, rel_model, loss_fn,
                    encoder_name, regularizer, max_len, num_negatives, lr,
                    use_scheduler, batch_size, emb_batch_size, eval_batch_size,
                    max_epochs, checkpoint, use_cached_ent_text, use_cached_rel_text,
                    _run: Run, _log: Logger, output_directory):
    _log.info(f'Encoder to be used: {encoder_name}')
    drop_stopwords = model in {'bert-bow', 'bert-dkrl',
                               'glove-bow', 'glove-dkrl'}

    prefix = 'ind-' if (dataset_setting != 'transductive') and model != 'transductive' else ''

    triples_file = f'dataset_creation/datasets/{dataset_setting}/{dataset}/{prefix}train.tsv'

    if device != torch.device('cpu'):
        num_devices = torch.cuda.device_count()
        if batch_size % num_devices != 0:
            raise ValueError(f'Batch size ({batch_size}) must be a multiple of'
                             f' the number of CUDA devices ({num_devices})')
        _log.info(f'CUDA devices used: {num_devices}')
    else:
        num_devices = 1
        _log.info('Training on CPU')

    if model == 'transductive':
        train_data = GraphDataset(triples_file, num_negatives,
                                  write_maps_file=True,
                                  num_devices=num_devices)
    else:
        if model.startswith('bert') or model == 'raild':
            tokenizer = BertTokenizer.from_pretrained(encoder_name)
        else:
            tokenizer = GloVeTokenizer('glove/glove.6B.300d-maps.pt')

        train_data = TextGraphDataset(triples_file, num_negatives,
                                      max_len, tokenizer, drop_stopwords,
                                      write_maps_file=False,
                                      use_cached_ent_text=use_cached_ent_text,
                                      use_cached_rel_text=use_cached_rel_text,
                                      num_devices=num_devices)

    train_loader = DataLoader(train_data, batch_size, shuffle=True,
                              collate_fn=train_data.collate_fn,
                              num_workers=0, drop_last=True)

    train_eval_loader = DataLoader(train_data, eval_batch_size)

    valid_data = GraphDataset(f'dataset_creation/datasets/{dataset_setting}/{dataset}/{prefix}dev.tsv')
    valid_loader = DataLoader(valid_data, eval_batch_size)

    test_data = GraphDataset(f'dataset_creation/datasets/{dataset_setting}/{dataset}/{prefix}test.tsv')
    test_loader = DataLoader(test_data, eval_batch_size)

    # Build graph with all triples to compute filtered metrics
    if (dataset_setting == 'tf-inductive') | (dataset_setting == 'f-inductive'):
        graph = None

        train_ent = set(train_data.entities.tolist())
        train_val_ent = set(valid_data.entities.tolist())
        train_val_test_ent = set(test_data.entities.tolist())
        val_new_ents = test_new_ents = None

        train_rel = set(train_data.relations.tolist())
        train_val_rel = set(valid_data.relations.tolist())
        train_val_test_rel = set(test_data.relations.tolist())
        #val_new_rels = test_new_rels = None

    else:
        graph = nx.MultiDiGraph()
        all_triples = torch.cat((train_data.triples,
                                 valid_data.triples,
                                 test_data.triples))
        graph.add_weighted_edges_from(all_triples.tolist())

        train_ent = set(train_data.entities.tolist())
        train_val_ent = set(valid_data.entities.tolist()).union(train_ent)
        train_val_test_ent = set(test_data.entities.tolist()).union(train_val_ent)
        val_new_ents = train_val_ent.difference(train_ent)
        test_new_ents = train_val_test_ent.difference(train_val_ent)

        train_rel = set(train_data.relations.tolist())
        train_val_rel = set(valid_data.relations.tolist()).union(train_rel)
        train_val_test_rel = set(test_data.relations.tolist()).union(train_val_rel)
        #val_new_rels = train_val_rel.difference(train_rel)
        #test_new_rels = train_val_test_rel.difference(val_new_rels)


    _run.log_scalar('num_train_entities', len(train_ent))

    train_ent = torch.tensor(list(train_ent))
    train_val_ent = torch.tensor(list(train_val_ent))
    train_val_test_ent = torch.tensor(list(train_val_test_ent))

    train_rel = torch.tensor(list(train_rel))
    train_val_rel = torch.tensor(list(train_val_rel))
    train_val_test_rel = torch.tensor(list(train_val_test_rel))

    model = utils.get_model(model, dim, rel_model, loss_fn,
                            len(train_val_test_ent), train_data.num_rels,
                            encoder_name, regularizer)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    if device != torch.device('cpu'):
        model = torch.nn.DataParallel(model).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * max_epochs
    if use_scheduler:
        warmup = int(0.2 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup,
                                                    num_training_steps=total_steps)
    best_valid_mrr = 0.0
    checkpoint_file = osp.join(output_directory, f'model-{_run._id}.pt')
    for epoch in range(1, max_epochs + 1):
        train_loss = 0
        for step, (text_tok_ent, text_mask_ent, text_tok_rel, text_mask_rel, rels, neg_idx) in enumerate(train_loader):
            loss = model(text_tok_ent=text_tok_ent, text_mask_ent=text_mask_ent,
                         text_tok_rel=text_tok_rel, text_mask_rel=text_mask_rel,
                         rels=rels, neg_idx=neg_idx).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            train_loss += loss.item()

            if step % int(0.05 * len(train_loader)) == 0:
                _log.info(f'Epoch {epoch}/{max_epochs} '
                          f'[{step}/{len(train_loader)}]: {loss.item():.6f}')
                _run.log_scalar('batch_loss', loss.item())

        _run.log_scalar('train_loss', train_loss / len(train_loader), epoch)

        if (dataset_setting != 'tf-inductive') & (dataset_setting != 'f-inductive'):
            _log.info('Evaluating on sample of training set')
            eval_link_prediction(model, train_eval_loader, train_data, train_ent,
                                 train_rel, epoch, emb_batch_size, prefix='train',
                                 max_num_batches=len(valid_loader), device=device)

        _log.info('Evaluating on validation set')
        val_mrr, _ = eval_link_prediction(model, valid_loader, train_data,
                                          train_val_ent, train_val_rel, epoch,
                                          emb_batch_size, prefix='valid', device=device)

        # Keep checkpoint of best performing model (based on raw MRR)
        if val_mrr > best_valid_mrr:
            best_valid_mrr = val_mrr
            torch.save(model.state_dict(), checkpoint_file)

    # Evaluate with best performing checkpoint
    if max_epochs > 0:
        model.load_state_dict(torch.load(checkpoint_file))


    if (dataset_setting == 'tf-inductive') | (dataset_setting == 'f-inductive'):
        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(valid_data.triples.tolist())

    _log.info('Evaluating on validation set (with filtering)')
    eval_link_prediction(model, valid_loader, train_data, train_val_ent, train_val_rel,
                         max_epochs + 1, emb_batch_size, prefix='valid',
                         filtering_graph=graph,
                         new_entities=val_new_ents, device=device)


    if (dataset_setting == 'tf-inductive') | (dataset_setting == 'f-inductive'):
        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(test_data.triples.tolist())

    _log.info('Evaluating on test set')
    _, ent_emb, rel_emb = eval_link_prediction(model, test_loader, train_data,
                                      train_val_test_ent, train_val_test_rel, max_epochs + 1,
                                      emb_batch_size, prefix='test',
                                      filtering_graph=graph,
                                      new_entities=test_new_ents,
                                      return_embeddings=True, device=device)

    # Save final entity embeddings obtained with trained encoder
    torch.save(model.module.state_dict(), checkpoint_file)
    torch.save(ent_emb, osp.join(output_directory, f'ent_emb-{_run._id}.pt'))
    torch.save(rel_emb, osp.join(output_directory, f'rel_emb-{_run._id}.pt'))
    torch.save(train_val_test_ent, osp.join(output_directory, f'ents-{_run._id}.pt'))



ex.run_commandline()
