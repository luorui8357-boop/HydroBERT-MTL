# Third party notices

## FT Transformer

The vendored FT-Transformer implementation is the archived author implementation of **Revisiting Deep Learning Models for Tabular Data**, Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov and Artem Babenko (NeurIPS 2021). Upstream: https://github.com/yandex-research/rtdl-revisiting-models . The original license is retained in `hydrobert/vendor/FTTransformer_LICENSE.txt`. The copied module is not otherwise modified; input width is configured by this project's wrapper.

## TabR

`hydrobert/vendor/tabr_model.py` contains the author Model class extracted from the archived **TabR: Tabular Deep Learning Meets Nearest Neighbors** implementation, Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov and Artem Babenko (ICLR 2024). Upstream: https://github.com/yandex-research/tabular-dl-tabr . The original license is retained in `hydrobert/vendor/TabR_LICENSE.txt`. The full upstream training application is excluded. Exact flat squared-L2 search replaces FAISS in a local adapter; tests cover gradient flow and repeated inference. Candidate labels remain training-only.

## SciBERT and TabPFN

SciBERT is initialized from `allenai/scibert_scivocab_uncased`: https://huggingface.co/allenai/scibert_scivocab_uncased . TabPFN uses the installed `tabpfn` package and separately obtained original v2 regression weights. Neither set of pretrained weights is redistributed in this source archive. Their upstream model/package license terms continue to apply.

The authors have not assigned a distribution license to the new project code in this preparation step. Choose and document the project license before public publication, while preserving the third-party notices and licenses.
