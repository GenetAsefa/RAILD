
import torch.nn as nn
from transformers import BertModel

from .inductive_lp import InductiveLinkPrediction


class BertEmbeddingsLP(InductiveLinkPrediction):
    """Using BERT for both entities and relations for Link Prediction (RAILD(w/o feat))."""
    def __init__(self, dim, rel_model, loss_fn, num_relations, encoder_name,
                 regularizer):
        super().__init__(dim, rel_model, loss_fn, num_relations, regularizer)
        self.encoder = BertModel.from_pretrained(encoder_name,
                                                 output_attentions=False,
                                                 output_hidden_states=False)
        hidden_size = self.encoder.config.hidden_size
        self.enc_linear = nn.Linear(hidden_size, self.dim, bias=False)

    def _encode_entity(self, text_tok, text_mask):
        # Extract BERT representation of [CLS] token
        embs = self.encoder(text_tok, text_mask)[0][:, 0]
        embs = self.enc_linear(embs)
        return embs

    def _encode_relation(self, text_tok, text_mask):
        # Extract BERT representation of [CLS] token
        embs = self.encoder(text_tok, text_mask)[0][:, 0]
        embs = self.enc_linear(embs)
        return embs
