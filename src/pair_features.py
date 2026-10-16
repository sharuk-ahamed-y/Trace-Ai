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


def build_115d_pair_feature(feat_a, feat_b):
    """
    Constructs the verified 115-dimensional pairwise feature vector:
    - 57 absolute-difference features (|feat_a - feat_b|)
    - 57 elementwise-product features (feat_a * feat_b)
    - 1 cosine-similarity feature (cos_sim(feat_a, feat_b))
    Total = 57 + 57 + 1 = 115 dimensions.

    Compatible with scikit-learn Pipeline(StandardScaler, MLPClassifier).
    """
    import numpy as np

    v_a = np.asarray(feat_a, dtype=np.float32).flatten()
    v_b = np.asarray(feat_b, dtype=np.float32).flatten()

    # Explicit validation: require exact 57 dimensions (17 forensic + 40 histogram)
    if len(v_a) != 57 or len(v_b) != 57:
        raise ValueError(
            f"[build_115d_pair_feature] Expected exact 57-dimensional features, "
            f"got len(feat_a)={len(v_a)} and len(feat_b)={len(v_b)}. "
            f"Features must comprise 17 scalar physical forensics + 40 histogram bins."
        )

    abs_diff = np.abs(v_a - v_b)
    elem_prod = v_a * v_b

    norm_a = np.linalg.norm(v_a) + 1e-8
    norm_b = np.linalg.norm(v_b) + 1e-8
    cos_sim = np.array([np.dot(v_a, v_b) / (norm_a * norm_b)], dtype=np.float32)

    return np.concatenate([abs_diff, elem_prod, cos_sim], axis=0).astype(np.float32)


def build_172d_direction_feature(feat_a, feat_b):
    """
    Constructs the verified 172-dimensional directional pairwise feature vector:
    - 57 features from feat_a (representation of image A)
    - 57 features from feat_b (representation of image B)
    - 57 signed difference features (feat_a - feat_b)
    - 1 cosine-similarity feature (cos_sim(feat_a, feat_b))
    Total = 57 + 57 + 57 + 1 = 172 dimensions.

    Compatible with scikit-learn direction_classifier.pkl (Pipeline: StandardScaler + MLPClassifier, input_dim=172).
    """
    import numpy as np

    v_a = np.asarray(feat_a, dtype=np.float32).flatten()
    v_b = np.asarray(feat_b, dtype=np.float32).flatten()

    # Explicit validation: require exact 57 dimensions
    if len(v_a) != 57 or len(v_b) != 57:
        raise ValueError(
            f"[build_172d_direction_feature] Expected exact 57-dimensional features, "
            f"got len(feat_a)={len(v_a)} and len(feat_b)={len(v_b)}. "
            f"Features must comprise 17 scalar physical forensics + 40 histogram bins."
        )

    signed_diff = v_a - v_b
    norm_a = np.linalg.norm(v_a) + 1e-8
    norm_b = np.linalg.norm(v_b) + 1e-8
    cos_sim = np.array([np.dot(v_a, v_b) / (norm_a * norm_b)], dtype=np.float32)

    return np.concatenate([v_a, v_b, signed_diff, cos_sim], axis=0).astype(np.float32)

