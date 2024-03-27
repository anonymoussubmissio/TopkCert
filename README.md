# TopkCert

This is the code for TopkCert.

## Environment

The code is implemented in Python==3.8, timm==0.9.10, torch==2.0.1.

## Demo

0. You may need to configure the location of datasets and checkpoints.

1. First, train base DL models. 

  ```python
  python train_drs.py --dataset gtsrb --ablation_type column --model vit_base_patch16_224 --ablation_size 19
  ```

  

2. Then, get the inference results of samples in the dataset from the DL models.

  ```python
  python certification_drs.py --dataset gtsrb --ablation_type column --model vit_base_patch16_224 --ablation_size 19
  ```


3. Finally, get the results of TopKCert.

   ```python
   python topkcert.py --dataset gtsrb --ablation_type column --model vit_base_patch16_224 --ablation_size 19
   ```

   

