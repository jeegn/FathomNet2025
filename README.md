Downloading Train/Test Data
python download.py ./dataset_train.json $SCRATCH/fathomNetData -n 5 

BioCLIP ZeroShot Classification
- Modified the BioCLIP zeroshot file to accomodate the dataset directory structure

sbatch scholar.sh python BioCLIP_zeroshot.py --train_data $SCRATCH/fathomNetData/ --exp bioclip-zero-shot

sbatch scholar.sh python BioCLIP_zeroshot.py --train_data $SCRATCH/fathomNetData/ --exp bioclip-zero-shot-openaitemps