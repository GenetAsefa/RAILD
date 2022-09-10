
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .inductive_lp import InductiveLinkPrediction

class WordEmbeddingsLP(InductiveLinkPrediction):
    """Description encoder with pretrained embeddings, obtained from BERT or a
    specified tensor file.
    """
    def __init__(self, rel_model, loss_fn, num_relations, regularizer,
                 dim=None, encoder_name=None, embeddings=None):
        if not encoder_name and not embeddings:
            raise ValueError('Must provided one of encoder_name or embeddings')

        if encoder_name is not None:
            encoder = BertModel.from_pretrained(encoder_name)
            embeddings = encoder.embeddings.word_embeddings
        else:
            emb_tensor = torch.load(embeddings)
            num_embeddings, embedding_dim = emb_tensor.shape
            embeddings = nn.Embedding(num_embeddings, embedding_dim)
            embeddings.weight.data = emb_tensor

        if dim is None:
            dim = embeddings.embedding_dim

        super().__init__(dim, rel_model, loss_fn, num_relations, regularizer)

        self.embeddings = embeddings

    def _encode_entity(self, text_tok, text_mask):
        raise NotImplementedError

    def _encode_relation(self, text_tok, text_mask):
        raise NotImplementedError


class BOW(WordEmbeddingsLP):
    """Bag-of-words (BOW) description encoder, with BERT low-level embeddings.
    """
    def _encode_entity(self, text_tok, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones_like(text_tok, dtype=torch.float)
        # Extract average of word embeddings
        embs = self.embeddings(text_tok)
        lengths = torch.sum(text_mask, dim=-1, keepdim=True)
        embs = torch.sum(text_mask.unsqueeze(dim=-1) * embs, dim=1)
        embs = embs / lengths

        return embs

    def _encode_relation(self, text_tok, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones_like(text_tok, dtype=torch.float)
        # Extract average of word embeddings
        embs = self.embeddings(text_tok)
        lengths = torch.sum(text_mask, dim=-1, keepdim=True)
        embs = torch.sum(text_mask.unsqueeze(dim=-1) * embs, dim=1)
        embs = embs / lengths

        return embs


class DKRL(WordEmbeddingsLP):
    """Description-Embodied Knowledge Representation Learning (DKRL) with CNN
    encoder, after
    Zuo, Yukun, et al. "Representation learning of knowledge graphs with
    entity attributes and multimedia descriptions."
    """

    def __init__(self, dim, rel_model, loss_fn, num_relations, regularizer,
                 encoder_name=None, embeddings=None):
        super().__init__(rel_model, loss_fn, num_relations, regularizer,
                         dim, encoder_name, embeddings)

        emb_dim = self.embeddings.embedding_dim
        self.conv1 = nn.Conv1d(emb_dim, self.dim, kernel_size=2)
        self.conv2 = nn.Conv1d(self.dim, self.dim, kernel_size=2)

    def _encode_entity(self, text_tok, text_mask):
        if text_mask is None:
            text_mask = torch.ones_like(text_tok, dtype=torch.float)
        # Extract word embeddings and mask padding
        embs = self.embeddings(text_tok) * text_mask.unsqueeze(dim=-1)

        # Reshape to (N, C, L)
        embs = embs.transpose(1, 2)
        text_mask = text_mask.unsqueeze(1)

        # Pass through CNN, adding padding for valid convolutions
        # and masking outputs due to padding
        embs = F.pad(embs, [0, 1])
        embs = self.conv1(embs)
        embs = embs * text_mask
        if embs.shape[2] >= 4:
            kernel_size = 4
        elif embs.shape[2] == 1:
            kernel_size = 1
        else:
            kernel_size = 2
        embs = F.max_pool1d(embs, kernel_size=kernel_size)
        text_mask = F.max_pool1d(text_mask, kernel_size=kernel_size)
        embs = torch.tanh(embs)
        embs = F.pad(embs, [0, 1])
        embs = self.conv2(embs)
        lengths = torch.sum(text_mask, dim=-1)
        embs = torch.sum(embs * text_mask, dim=-1) / lengths
        embs = torch.tanh(embs)

        return embs

    def _encode_relation(self, text_tok, text_mask):
        if text_mask is None:
            text_mask = torch.ones_like(text_tok, dtype=torch.float)
        # Extract word embeddings and mask padding
        embs = self.embeddings(text_tok) * text_mask.unsqueeze(dim=-1)

        # Reshape to (N, C, L)
        embs = embs.transpose(1, 2)
        text_mask = text_mask.unsqueeze(1)

        # Pass through CNN, adding padding for valid convolutions
        # and masking outputs due to padding
        embs = F.pad(embs, [0, 1])
        embs = self.conv1(embs)
        embs = embs * text_mask
        if embs.shape[2] >= 4:
            kernel_size = 4
        elif embs.shape[2] == 1:
            kernel_size = 1
        else:
            kernel_size = 2
        embs = F.max_pool1d(embs, kernel_size=kernel_size)
        text_mask = F.max_pool1d(text_mask, kernel_size=kernel_size)
        embs = torch.tanh(embs)
        embs = F.pad(embs, [0, 1])
        embs = self.conv2(embs)
        lengths = torch.sum(text_mask, dim=-1)
        embs = torch.sum(embs * text_mask, dim=-1) / lengths
        embs = torch.tanh(embs)

        return embs
