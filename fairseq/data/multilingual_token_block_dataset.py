# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.data.token_block_dataset import TokenBlockDataset


class MultilingualTokenBlockDataset(TokenBlockDataset):
    """Break a Dataset of tokens into blocks.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
    """

    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        lang_id,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
    ):
        super().__init__(
            dataset,
            sizes,
            block_size,
            pad,
            eos,
            break_mode=break_mode,
            include_targets=include_targets,
            document_sep_len=document_sep_len,
        )
        self.lang_id = lang_id

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

        buffer = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )

        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length
        item = buffer[s:e]

        if self.include_targets:
            # *target* is the original sentence (=item)
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            if s == 0:
                source = torch.cat([item.new([self.lang_id]), buffer[0 : e - 1]])
                past_target = torch.cat(
                    [item.new([self.pad, self.eos]), buffer[0 : e - 2]]
                )
            else:
                source = buffer[s - 1 : e - 1]
                if s == 1:
                    past_target = torch.cat([item.new([self.eos]), buffer[0 : e - 2]])
                else:
                    past_target = buffer[s - 2 : e - 2]

            return source, item, past_target

        return item
