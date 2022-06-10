from hodor.config import cfg
from hodor.modelling.backbone import SwinTransformerFPNTiny
from hodor.modelling.encoder import Encoder
from hodor.modelling.decoder import Decoder

import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = SwinTransformerFPNTiny(
            pretrained=True,
            n_dims=256
        )

        self.encoder = Encoder(
            n_dims=256, n_layers=cfg.MODEL.ENCODER_LAYERS,
            pre_normalize=cfg.MODEL.TRANSFORMER_PRE_NORMALIZE,
            n_heads=cfg.MODEL.NUM_ATTN_HEADS,
            n_bg_queries=cfg.MODEL.NUM_BG_DESCRIPTORS,
            pos_encodings=cfg.MODEL.POS_ENCODINGS_IN_FMQ
        )

        self.decoder = Decoder(
            n_dims=256, n_layers=cfg.MODEL.DECODER_LAYERS,
            pre_normalize=cfg.MODEL.TRANSFORMER_PRE_NORMALIZE,
            n_heads=cfg.MODEL.NUM_ATTN_HEADS,
            pos_encodings=cfg.MODEL.POS_ENCODINGS_IN_FQM,
            n_catch_all_queries=cfg.MODEL.NUM_CATCH_ALL_QUERIES
        )

    @property
    def num_catch_all_queries(self):
        return cfg.MODEL.NUM_CATCH_ALL_QUERIES

    @property
    def num_bg_queries(self):
        return cfg.MODEL.NUM_BG_DESCRIPTORS

    @property
    def pos_embed_fg_descriptors(self):
        return getattr(self.encoder, "fg_query_embed", None)

    @property
    def pos_embed_bg_descriptors(self):
        return getattr(self.encoder, "bg_query_embed", None)

    @property
    def encoder_mask_scale(self):
        return self.encoder.mask_scale
