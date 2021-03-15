# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@dataclass
class HuggingfaceTokenizerConfig(FairseqDataclass):
    code: str = field(default="gpt2", metadata={"help": "huggingface code"})

@register_tokenizer("huggingface", dataclass=HuggingfaceTokenizerConfig)
class HuggingfaceTokenizer(object):
    def __init__(self, cfg: HuggingfaceTokenizerConfig):
        self.cfg = cfg

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.code)
        except ImportError:
            raise ImportError(
                "Wrong with AutoTokenizer in Transformers of Huggingface"
            )

    def encode(self, x: str) -> str:
        return ' '.join(self.tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        return self.tokenizer.convert_tokens_to_string(x.split(' '))
