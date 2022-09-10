import torch
import torch.nn as nn
import sys
#sys.path.append("..")

#__all__=["transe_score", "distmult_score", "complex_score", "simple_score"]

def transe_score(heads, tails, rels):
    return -torch.norm(heads + rels - tails, dim=-1, p=1)


def distmult_score(heads, tails, rels):
    return torch.sum(heads * rels * tails, dim=-1)


def complex_score(heads, tails, rels):
    heads_re, heads_im = torch.chunk(heads, chunks=2, dim=-1)
    tails_re, tails_im = torch.chunk(tails, chunks=2, dim=-1)
    rels_re, rels_im = torch.chunk(rels, chunks=2, dim=-1)

    return torch.sum(rels_re * heads_re * tails_re +
                     rels_re * heads_im * tails_im +
                     rels_im * heads_re * tails_im -
                     rels_im * heads_im * tails_re,
                     dim=-1)


def simple_score(heads, tails, rels):
    heads_h, heads_t = torch.chunk(heads, chunks=2, dim=-1)
    tails_h, tails_t = torch.chunk(tails, chunks=2, dim=-1)
    rel_a, rel_b = torch.chunk(rels, chunks=2, dim=-1)

    return torch.sum(heads_h * rel_a * tails_t +
                     tails_h * rel_b * heads_t, dim=-1) / 2
