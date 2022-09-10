import torch
import torch.nn as nn
from transformers import BertModel

from .inductive_lp import InductiveLinkPrediction


class BertCombinedLP(InductiveLinkPrediction):
    """Encoding relaitons using both BERT and structural features while encding entities using Bert for link Prediction (RAILD)."""
    def __init__(self, dim, rel_model, loss_fn, num_relations, encoder_name,
                 regularizer):
        super().__init__(dim, rel_model, loss_fn, num_relations, regularizer)
        self.encoder = BertModel.from_pretrained(encoder_name,
                                                 output_attentions=False,
                                                 output_hidden_states=False)
        hidden_size = self.encoder.config.hidden_size
        dimension = self.dim // 2
        print("dimension:", dimension)
        self.enc_linear = nn.Linear(hidden_size, dimension, bias=False)

    def _encode_entity(self, text_tok, text_mask):
        # Extract BERT representation of [CLS] token
        embs = self.encoder(text_tok, text_mask)[0][:, 0]
        embs = self.enc_linear(embs)
        embs_concat = torch.cat((embs, embs), 1)
        return embs_concat

    def _encode_relation(self, text_tok, text_mask, node2vec_embs):
        # Extract BERT representation of [CLS] token
        embs_text = self.encoder(text_tok, text_mask)[0][:, 0]
        embs_text = self.enc_linear(embs_text)
        embs_node2vec = self.enc_linear(node2vec_embs)
        embs_concat = torch.cat((embs_text, embs_node2vec), 1)

        return embs_concat
