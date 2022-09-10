
from .base import LinkPrediction


class InductiveLinkPrediction(LinkPrediction):
    """Description-based Link Prediction (DLP)."""
    def _encode_entity(self, text_tok, text_mask):
        raise NotImplementedError

    def _encode_relation(self, text_tok, text_mask, *args, **kwargs):
        raise NotImplementedError

    def forward(self, text_tok_ent=None, text_mask_ent=None, text_tok_rel=None, text_mask_rel=None, relation_features=None, rels=None, neg_idx=None):

        if neg_idx is None:
            # Forward is being used to compute entity and or relation embeddings only
            if text_tok_ent is not None and text_mask_ent is not None:
                batch_size, _, num_text_tokens = text_tok_ent.shape

                #Encode text into an entity representation from its description
                ent_embs = self.encode_entity(text_tok_ent.view(-1, num_text_tokens),
                                       text_mask_ent.view(-1, num_text_tokens))
                out =  ent_embs

            elif text_tok_rel is not None and text_mask_rel is not None and relation_features is not None: ## i.e., BertCombinedLP model:
                #encode relations using their textual descriptions and also rel-rel-net features.
                _, _, num_text_tokens_rel= text_tok_rel.shape
                rel_embs = self.encode_relation(text_tok_rel.view(-1, num_text_tokens_rel),
                                       text_mask_rel.view(-1, num_text_tokens_rel),
                                       relation_features)
                out =  rel_embs

            elif text_tok_rel is not None and text_mask_rel is not None:

                _, _, num_text_tokens_rel= text_tok_rel.shape

                #encode relations using only their textual description
                rel_embs = self.encode_relation(text_tok_rel.view(-1, num_text_tokens_rel),
                                       text_mask_rel.view(-1, num_text_tokens_rel))
                out =  rel_embs
            else:
                #encode relations using their rel-rel-net features
                rel_embs = self.encode_relation(relation_features)
                out =  rel_embs

        else:
            #Encode text into an entity representation from its description
            batch_size, _, num_text_tokens = text_tok_ent.shape

            ent_embs = self.encode_entity(text_tok_ent.view(-1, num_text_tokens),
                                   text_mask_ent.view(-1, num_text_tokens))

            if text_tok_rel is not None and text_mask_rel is not None and relation_features is not None: ## i.e., BertCombinedLP model
                #encode relations using thier textual descriptions and also rel-rel-net features.
                _, _, num_text_tokens_rel= text_tok_rel.shape
                rel_embs = self.encode_relation(text_tok_rel.view(-1, num_text_tokens_rel),
                                       text_mask_rel.view(-1, num_text_tokens_rel),
                                       relation_features)

            elif text_tok_rel is not None and text_mask_rel is not None:
                #encode relations using only their textual descriptions
                _, _, num_text_tokens_rel= text_tok_rel.shape
                rel_embs = self.encode_relation(text_tok_rel.view(-1, num_text_tokens_rel),
                                       text_mask_rel.view(-1, num_text_tokens_rel))

            else:
                #encode relations using their rel-rel-net features
                rel_embs = self.encode_relation(relation_features)

            ent_embs = ent_embs.view(batch_size, 2, -1)

            rel_embs = rel_embs.view(batch_size, 1, -1)

            out = self.compute_loss(ent_embs, rel_embs, neg_idx)

        return out
