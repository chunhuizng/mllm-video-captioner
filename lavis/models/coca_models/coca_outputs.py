from dataclasses import dataclass

from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class CoCaOutputFeatures(ModelOutput):
    """
    Data class of features from AlbefFeatureExtractor.
    Args:
        image_embeds: `torch.FloatTensor` of shape `(batch_size, 1, embed_dim)`, `optional`
        image_features: `torch.FloatTensor` of shape `(batch_size, 1, feature_dim)`, `optional`
        text_embeds: `torch.FloatTensor` of shape `(batch_size, 1, embed_dim)`, `optional`
        text_features: `torch.FloatTensor` of shape `(batch_size, 1, feature_dim)`, `optional`
    """

    image_embeds: Optional[torch.FloatTensor] = None
    image_embeds_proj: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None


@dataclass
class CoCaOutput(ModelOutput):
    intermediate_output: Optional[CoCaOutputFeatures] = None

    logit_scale_exp: Optional[torch.FloatTensor] = None

    loss_clip: Optional[torch.FloatTensor] = None ### contrastive loss
    loss_caption: Optional[torch.FloatTensor] = None ### caption loss
    loss: Optional[torch.FloatTensor] = None