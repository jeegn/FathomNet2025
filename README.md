Downloading Train/Test Data
python download.py ./dataset_test/dataset_test.json outputdir -n 5 


BioCLIP ZeroShot Classification
- Modified the BioCLIP zeroshot file to accomodate the dataset directory structure

python BioCLIP_zeroshot.py --train_data ./dataset_train --exp bioclip-zero-shot