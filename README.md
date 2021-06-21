# CLIP inference for ImageNet and train other models using soft and hard labels from CLIP

This repo contains CLIP model inference for ImageNet dataset and using the resulting hard and soft labels one can train any other models using thse labels.

## Quick Start

Clone the repository and install the requirements using pip

```
git clone https://github.com/kantharajucn/clip-imagenet-evaluation

pip install -r requirements.txt

```


## Running CLIP inference on ImageNet

```
python3 clip_imagenet_inference.py --data-dir /path/to/dataset/dir --num-workers 25 --batch-size 40

```

Inference results will be stored into files `soft_labels.json` and `hard_labels.json` in the current directory. 

## Run model training using CLIP hard and soft labels

You can train any model from torch model zoo on both `Soft Labels` and `Hard Labels`. After the successful training of the model, checkpoints will be stored in `checkpoints` directory.

To train using soft labels

```
train.py /path/to/imagenet -a resnet50 --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --label-type soft_labels

```

To train using hard labels

```
train.py /path/to/imagenet -a resnet50 --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 16  --label-type hard_labels

```


## Evaluate model trained using CLIP hard and Soft Labels

After training any `pytorch zoo model` using CLIP labels, you can evaluate the model on ImageNet with original ImageNet labels or CLIP labels using the below script.

### Evaluate on ImageNet labels

```
test.py /path/to/imagenet -a resnet50 --evaluate --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --checkpoint path/to/hard_labels_checkpoint.pth
```


### Evaluate on CLIP hard labels
```
test.py /path/to/imagenet -a resnet50 --evaluate --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --label-type hard_labels --labels path/to/CLIP/IN_val_clip_hard_labels.json --checkpoint path/to/hard_labels_checkpoint.pth.tar
```

### Evaluate on CLIP soft labels
```
test.py /path/to/imagenet -a resnet50 --evaluate --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --label-type soft_labels --labels path/to/IN_val_clip_soft_labels.json --checkpoint path/to/soft_labels_checkpoint.pth.tar
```


## Running training and evaluation on Slurm

Use the scripts `rain.sh` and `test.sh` to run training and testing the models on SLURM cluster.

## License

1. CLIP model is taken from https://github.com/openai/CLIP, please adhere to the license of this repository.
