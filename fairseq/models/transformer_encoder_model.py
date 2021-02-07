# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Optional

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerEncoder
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II


DEFAULT_max_source_positions = 1024


@dataclass
class TransformerEncoderModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    encoder_embed_dim: int = field(
        default=512, metadata={"help": "encoder embedding dimension"}
    )
    encoder_output_dim: int = field(
        default=512, metadata={"help": "encoder output dimension"}
    )
    encoder_input_dim: int = field(
        default=512, metadata={"help": "encoder input dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(default=6, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    no_encoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last encoder block"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_encoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share encoder input and output embeddings"}
    )
    character_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, uses character embedding convolutions to produce token embeddings"
        },
    )
    character_filters: str = field(
        default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
        metadata={"help": "size of character embeddings"},
    )
    character_embedding_dim: int = field(
        default=4, metadata={"help": "size of character embeddings"}
    )
    char_embedder_highway_layers: int = field(
        default=2,
        metadata={"help": "number of highway layers for character token embeddder"},
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_input_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    adaptive_input_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive input cutoff points."},
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    encoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the encoder"},
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for encoder"}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    # TODO common var add to parent
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_source_positions: Optional[int] = II("task.max_source_positions")
    tpu: bool = II("common.tpu")


@register_model("transformer_encoder_model", dataclass=TransformerEncoderModelConfig)
class TransformerEncoderModel(FairseqEncoderModel):

    def __init__(self, encoder):
        super().__init__(encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_encoder_model_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = getattr(
                args, "tokens_per_sample", DEFAULT_max_source_positions
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.encoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.encoder_input_dim,
                args.adaptive_input_factor,
                args.encoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.encoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.encoder_input_dim == args.encoder_output_dim

        encoder = TransformerEncoder(
            args, task.source_dictionary, embed_tokens
        )
        if args.share_encoder_input_output_embed:
            encoder.output_projection = nn.Linear(
                encoder.embed_tokens.weight.shape[1],
                encoder.embed_tokens.weight.shape[0],
                bias=False,
            )
            encoder.output_projection.weight = encoder.embed_tokens.weight
        else:
            encoder.output_projection = nn.Linear(
                encoder.output_embed_dim, encoder.embed_tokens.weight.shape[0], bias=False
            )
            nn.init.normal_(
                encoder.output_projection.weight, mean=0, std=encoder.output_embed_dim ** -0.5
            )
        return cls(encoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["encoder_out"][0]
        encoder_out = self.encoder.output_projection(encoder_out)

        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)


def base_encoder_model_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_encoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "encoder_final_norm"):
        args.no_encoder_final_norm = not args.encoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.encoder_output_dim = getattr(
        args, "encoder_output_dim", args.encoder_embed_dim
    )
    args.encoder_input_dim = getattr(args, "encoder_input_dim", args.encoder_embed_dim)

    # Model training is not stable without this
    args.encoder_normalize_before = True
    args.no_encoder_final_norm = getattr(args, "no_encoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("transformer_encoder_model", "transformer_encoder_model_6l_16h_1024")
def transformer_encoder_model_6l_16h_1024(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_encoder_model_architecture(args)

