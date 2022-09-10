
import torch.nn as nn
from transformers import BertModel, DistilBertModel

from .inductive_lp import InductiveLinkPrediction


class BertNode2Vec(InductiveLinkPrediction):
    """Using BERT for entities and graph-based features for relations for Link Prediction (RAILD(w/o txt))."""
    def __init__(self, dim, rel_model, loss_fn, num_relations, encoder_name,
                 regularizer):
        super().__init__(dim, rel_model, loss_fn, num_relations, regularizer)
        self.encoder = BertModel.from_pretrained(encoder_name,
                                                 output_attentions=False,
                                                 output_hidden_states=False)
        hidden_size = self.encoder.config.hidden_size
        self.enc_linear = nn.Linear(hidden_size, self.dim, bias=False)

    def _encode_entity(self, text_tok, text_mask):
        embs = self.encoder(text_tok, text_mask)[0][:, 0]
        embs = self.enc_linear(embs)
        return embs

    def _encode_relation(self, node2vec_embs, *args, **kwargs):
        embs = self.enc_linear(node2vec_embs)
        return embs


    def _encode_relation(self, node2vec_embs, *args, **kwargs):
        embs = self.enc_linear_rel(node2vec_embs)
        return embs
