"""
pair_features.py — Build pair-wise feature vectors for the MLP classifiers.

Feature vector for a (A, B) pair:
    [emb_A | emb_B | |emb_A - emb_B| | cosine_similarity(A, B)]
Total dim = 768 + 768 + 768 + 1 = 2305
"""

import torch
import torch.nn.functional as F
from typing import Tuple


PAIR_FEATURE_DIM = 2305   # 3 * 768 + 1


def build_pair_feature(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
) -> torch.Tensor:
    """
    Build a single pair-feature vector from two L2-normalised embeddings.

    Parameters
    ----------
    emb_a, emb_b : (768,) CPU tensors

    Returns
    -------
    feature : (2305,) CPU tensor
    """
    emb_a = emb_a.float()
    emb_b = emb_b.float()

    abs_diff = (emb_a - emb_b).abs()                         # (768,)
    cos_sim  = F.cosine_similarity(emb_a.unsqueeze(0),
                                   emb_b.unsqueeze(0)).cpu() # (1,)

    feature = torch.cat([emb_a, emb_b, abs_diff, cos_sim])  # (2305,)
    return feature


def build_pair_feature_batch(
    embs_a: torch.Tensor,
    embs_b: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorised version: build features for B pairs at once.

    Parameters
    ----------
    embs_a, embs_b : (B, 768) tensors

    Returns
    -------
    features : (B, 2305) tensor
    """
    embs_a = embs_a.float()
    embs_b = embs_b.float()

    abs_diff = (embs_a - embs_b).abs()                             # (B, 768)
    cos_sim  = F.cosine_similarity(embs_a, embs_b).unsqueeze(1)   # (B, 1)

    features = torch.cat([embs_a, embs_b, abs_diff, cos_sim], dim=1)  # (B, 2305)
    return features
