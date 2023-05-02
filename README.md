# Miracle
Codes for SIGIR 2023 Full Paper-**Towards Multi-Interest Pre-training with Sparse Capsule Network**

## Step1. Download the Datasets:
  Firstly, create a folder named "dataset". Then, within the "dataset" folder, create two subfolders named "Ratings" and "Metadata" respectively. Then you can donwload the dataset from [Amazon](https://nijianmo.github.io/amazon/index.html). The .csv files should be placed in the "Ratings" folder, while meta_\*.json.gz files should be placed in the "Metadata" folder. Thanks to [UniSRec](https://github.com/RUCAIBox/UniSRec) for providing another link to download datasets from cloud disks
 
## Step2. Process Datasets:
  You only needs *run process_amazon.py* to process the raw datasets if you placed the dataset as required in Step1.
  
## Step3. Run Pretrain:
  You can pretrain the model by *run main.py --dataset=FHCKM --gpu_id=0 --save_step=1 --epoch=3 --stage=pretrain --train_batch_size=2048*
  Note: the train_batch_size hyperparameter is important, if you modify this hyperparameter, you may get unexpected results due to overfitting. The FHCKM means the mixed pretrain dataset.
 
## Step4. Run Downstream:
  You can test the model by *run main.py --dataset=Scientific --gpu_id=0--epoch=100 --stage=trans --train_batch_size=1024 --load_model_path='{pretrain_model_path in Step3}'*
  Note: the load_model_path parameter is printed in Step3, please use the checkpoint ending with "pretrain-2.pkl"(which pretrain for 3 epochs), such as "./saved_model/2-13-50-10/pretrain-2.pkl"


I have run the entire code on my computer to ensure that it can be used, if you encounter problems, you can contact me by email *tangzuoli@whu.edu.cn*
### Acknowledgement
Thanks to [UniSRec](https://github.com/RUCAIBox/UniSRec) for its open source code and datasets.

