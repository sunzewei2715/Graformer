# GraftTransformer

The repository for the paper: **Multilingual Translation via Grafting Pre-trained NLP Models**

**GraftTransformer** (also named BridgeTransformer in the code) is a sequence-to-sequence model mainly for Neural Machine Translation. We improve the multilingual translation by taking advantage of pre-trained NLP models, including pre-trained encoder (**BERT**) and pre-trained decoder (**GPT**). The code is based on [Fairseq](https://github.com/pytorch/fairseq).

### examples
You can start with *run/run_arnold.sh*, with some minor modification. The corresponding scripts represent:
```
train a pre-trained BERT:
    run_arnold_multilingual_masked_lm_6e6d.sh

train a pre-trained GPT:
    run_arnold_multilingual_lm_6e6d.sh

train a GraftTransformer:    
    run_arnold_multilingual_graft_transformer_12e12d_ted.sh

inference from GraftTransformer:
    run_arnold_multilingual_graft_inference_ted.sh
    
```

# Citation

Please cite as:

``` bibtex
@inproceedings{sun2021mulilingual,
    title = "Multilingual Translation via Grafting Pre-trained NLP Models",
    author = "Sun, Zewei and Wang, Mingxuan and Li, Lei",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021"
}
```
