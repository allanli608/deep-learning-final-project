1. `conda env create -f environment_local.yml`
2. `conda env update -f environment_local.yml --prune`
3. `conda activate deeplearning-local`
4. `python ./src/data_collection/wnc_download.py`
5. run `wnc_download.py` and then `preprocess.py`