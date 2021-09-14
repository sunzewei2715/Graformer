# Graformer

The repository for the paper: [Multilingual Translation via Grafting Pre-trained Language Models](https://arxiv.org/abs/2109.05256)

**Graformer** (also named BridgeTransformer in the code) is a sequence-to-sequence model mainly for Neural Machine Translation. We improve the multilingual translation by taking advantage of pre-trained (masked) language models, including pre-trained encoder (**BERT**) and pre-trained decoder (**GPT**). The code is based on [Fairseq](https://github.com/pytorch/fairseq).

## Examples
You can start with *run/run.sh*, with some minor modification. The corresponding scripts represent:
```
train a pre-trained BERT:
    run_arnold_multilingual_masked_lm_6e6d.sh

train a pre-trained GPT:
    run_arnold_multilingual_lm_6e6d.sh

train a Graformer:
    run_arnold_multilingual_graft_transformer_12e12d_ted.sh

inference from Graformer:
    run_arnold_multilingual_graft_inference_ted.sh
    
```

## Released Models
We release our pre-trained **mBERT** and **mGPT**, along with the trained **Graformer** model in [here](https://drive.google.com/drive/folders/1WBleOk_sT-D06bxug_Pop77u3ZDx_mZb?usp=sharing).

## Tensorflow Version
We will provide the tensorflow version in [Neurst](https://github.com/bytedance/neurst), a popular toolkit for sequence processing.

## Citation

Please cite as:

``` bibtex
@inproceedings{sun2021mulilingual,
    title = "Multilingual Translation via Grafting Pre-trained Language Models",
    author = "Sun, Zewei and Wang, Mingxuan and Li, Lei",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021"
}
```

## Contact

If you have any questions, please feel free to contact me: sunzewei.v@bytedance.com