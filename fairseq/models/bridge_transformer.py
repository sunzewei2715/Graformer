# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoder,
    base_architecture,
    register_model,
)
from fairseq.models import register_model_architecture
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


@register_model("bridge_transformer_model")
class BridgeTransformerModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.freeze_params(args)

    def freeze_params(self, args):
        freeze_pattern = re.compile(args.freeze_params)
        for name, parameter in self.named_parameters():
            if freeze_pattern.search(name):
                parameter.requires_grad = False
                logger.info(f"Freeze: {name}")

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--no-encoder-attn-layers', type=str, metavar='D', default="",
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument('--freeze-params', type=str, metavar='D', default="",
                            help='regular expression of parameters that need to be frozen')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return BridgeTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def load_state_dict(
        self,
        state_dict,
        strict=False,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=False)


class BridgeTransformerDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        no_encoder_attn_layers = args.no_encoder_attn_layers.split(',')
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn=True)
                if str(layer) in no_encoder_attn_layers else self.build_decoder_layer(args, no_encoder_attn=False)
                for layer in range(args.decoder_layers)
            ]
        )


@register_model_architecture("bridge_transformer_model", "bridge_transformer")
def bridge_transformer_model(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)
