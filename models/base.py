import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
#sys.path.append("..")

import scoring, losses


class LinkPrediction(nn.Module):
    """A general link prediction model with a lookup table for relation
    embeddings."""
    def __init__(self, dim, rel_model, loss_fn, num_relations, regularizer):
        super().__init__()
        self.dim = dim
        self.normalize_embs = False
        self.regularizer = regularizer

        if rel_model == 'transe':
            self.score_fn = scoring.transe_score
            self.normalize_embs = True
        elif rel_model == 'distmult':
            self.score_fn = scoring.distmult_score
        elif rel_model == 'complex':
            self.score_fn = scoring.complex_score
        elif rel_model == 'simple':
            self.score_fn = scoring.simple_score
        else:
            raise ValueError(f'Unknown relational model {rel_model}.')

        if loss_fn == 'margin':
            self.loss_fn = losses.margin_loss
        elif loss_fn == 'nll':
            self.loss_fn = losses.nll_loss
        else:
            raise ValueError(f'Unkown loss function {loss_fn}')

    def encode_entity(self, *args, **kwargs):
        ent_emb = self._encode_entity(*args, **kwargs)
        if self.normalize_embs:
            ent_emb = F.normalize(ent_emb, dim=-1)

        return ent_emb

    def encode_relation(self, *args, **kwargs):
        rel_emb = self._encode_relation(*args, **kwargs)
        if self.normalize_embs:
            rel_emb = F.normalize(rel_emb, dim=-1)

        return rel_emb

    def _encode_entity(self, *args, **kwargs):
        raise NotImplementedError

    def _encode_relation(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, ent_embs, rel_embs, neg_idx):
        batch_size = ent_embs.shape[0]

        # Scores for positive samples
        #rels = self.rel_emb(rels)
        heads, tails = torch.chunk(ent_embs, chunks=2, dim=1)

        pos_scores = self.score_fn(heads, tails, rel_embs)

        if self.regularizer > 0:
            #reg_loss = self.regularizer * l2_regularization(heads, tails, rels)
            reg_loss = self.regularizer * l2_regularization(heads, tails, rel_embs)
        else:
            reg_loss = 0

        # Scores for negative samples
        neg_embs = ent_embs.view(batch_size * 2, -1)[neg_idx]
        heads, tails = torch.chunk(neg_embs, chunks=2, dim=2)
        #neg_scores = self.score_fn(heads.squeeze(), tails.squeeze(), rels)
        neg_scores = self.score_fn(heads.squeeze(), tails.squeeze(), rel_embs)

        model_loss = self.loss_fn(pos_scores, neg_scores)
        return model_loss + reg_loss


def l2_regularization(heads, tails, rels):
    reg_loss = 0.0
    for tensor in (heads, tails, rels):
        reg_loss += torch.mean(tensor ** 2)

    return reg_loss / 3.0
