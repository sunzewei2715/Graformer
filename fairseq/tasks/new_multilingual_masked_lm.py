# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from fairseq import utils
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    RawLabelDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    encoders,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.multilingual.multilingual_utils import LangTokStyle, augment_dictionary, get_lang_tok
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
logger = logging.getLogger(__name__)

@dataclass
class NewMultiLingualMaskedLMConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    multilang_sampling_alpha: float = field(
        default=1.0,
        metadata={"help": "smoothing alpha for sample rations across multiple datasets"},
    )
    langs: str = field(
        default="",
        metadata={
            "help": "comma-separated list of languages"
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    lang_tok_style: str = field(
        default=LangTokStyle.multilingual.value,
        metadata={
            "help": "language token styles"
            'e.g., LangTokStyle.multilingual.value / LangTokStyle.mbart.value'
        },
    )
    encoder_langtok: Optional[str] = field(
        default=None,
        metadata={"help": "whether to add language token at the beginning: None/src"},
    )
    replace_mask_with_bos: bool = field(
        default=False,
        metadata={"help": "whether to replace <mask> with <s>"},
    )
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")


@register_task("new_multilingual_masked_lm", dataclass=NewMultiLingualMaskedLMConfig)
class NewMultiLingualMaskedLMTask(LegacyFairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        lang_list = args.langs.split(',')
        augment_dictionary(dictionary, lang_list, args.lang_tok_style)
        self.lang_tok_style = args.lang_tok_style
        self.encoder_langtok = args.encoder_langtok

        # add mask token
        if args.replace_mask_with_bos:
            self.mask_idx = dictionary.bos_index
        else:
            self.mask_idx = dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def _get_whole_word_mask(self):
        # create masked input and targets
        if self.args.mask_whole_words:
            bpe = encoders.build_bpe(self.args)
            if bpe is not None:

                def is_beginning_of_word(i):
                    if i < self.source_dictionary.nspecial:
                        # special elements are always considered beginnings
                        return True
                    tok = self.source_dictionary[i]
                    if tok.startswith("madeupword"):
                        return True
                    try:
                        return bpe.is_beginning_of_word(tok)
                    except ValueError:
                        return True

                mask_whole_words = torch.ByteTensor(
                    list(map(is_beginning_of_word, range(len(self.source_dictionary))))
                )
        else:
            mask_whole_words = None
        return mask_whole_words

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        languages = sorted(
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        )

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: ", {lang: id for id, lang in enumerate(languages)}
        )

        mask_whole_words = self._get_whole_word_mask()
        lang_datasets = []
        for lang_id, language in enumerate(languages):
            split_path = os.path.join(data_path, language, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode=self.args.sample_break_mode,
            )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            lang_token = get_lang_tok(language, self.lang_tok_style)
            lang_id = self.dictionary.index(lang_token)
            if self.encoder_langtok is None:
                beginning_token = None
            elif self.encoder_langtok == "bos":
                # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
                beginning_token = self.source_dictionary.bos()
            elif self.encoder_langtok == "src":
                beginning_token = lang_id
            else:
                raise Exception("wrong indicator of beginning token!")
            dataset = PrependTokenDataset(dataset, beginning_token)

            src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
                dataset,
                self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
            )

            lang_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "target": PadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                    "lang_id": RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
                },
                sizes=[src_dataset.sizes],
            )
            lang_datasets.append(lang_dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            "loaded total {} blocks for all languages".format(
                dataset_lengths.sum(),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: ",
                {
                    lang: "{0:.4f}".format(sample_probs[id])
                    for id, lang in enumerate(languages)
                },
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: ",
                {
                    lang: "{0:.2f}".format(size_ratio[id])
                    for id, lang in enumerate(languages)
                },
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(resampled_lang_datasets)
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + "_" + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            # [TODO]: This is hacky for now to print validation ppl for each
            # language individually. Maybe need task API changes to allow it
            # in more generic ways.
            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ",".join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
