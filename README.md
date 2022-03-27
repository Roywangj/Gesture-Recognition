# USAGE OF GESTURE_RECONGNITION


Test pretained model:
```shell
CUDA_VISIBLE_DEVICES=0,1 python test.py --pretrained_model_path /ResNet101-pretain/model/
```


End to end train:
```shell
#  Preprocess data
python split_data.py

#  Train
CUDA_VISIBLE_DEVICES=0,1 python train.py

#  Test
CUDA_VISIBLE_DEVICES=0,1 python test.py --pretrained_model_path /ResNet101-xxxxxxxxxxxx/model/
```

logs are in checkpoints/ResNet101-xxxxxxxxxxxx/model/out.txt, testout.txt


---
@author: Jie Wang & Peifu Liu & Yutong Sun

Email: jwang991020@gmail.com