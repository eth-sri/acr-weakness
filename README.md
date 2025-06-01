## Repository overview
Our code and repository is largely based on several previous works ([Cohen et al (2019)](https://github.com/locuslab/smoothing), [Salman et al (2019)](https://github.com/Hadisalman/smoothing-adversarial), [Jeong and Shin (2020)](https://github.com/jh-jeong/smoothing-consistency), [Jeong et al (2021)](https://github.com/jh-jeong/smoothmix), and [Jeong and Shin (2023)](https://github.com/alinlab/smoothing-catrs)). Major parts of the repo are:

* ```code/```: All training, testing, certification and analyze related code. Our main training code is ```code/train.py```.
* ```dataset_cache/```: Cache of CIFAR-10 dataset.
* ```logs/```: Logs and checkpoints during training.
* ```p_correct/```: The evaluated p_A data during training.
* ```scripts/```: Scripts to run our experiments.
* ```test/```: The ceritification results.

## Environmental setup
```
conda create -n acr_exp python=3.9
conda activate acr_exp

conda install scipy pandas statsmodels matplotlib seaborn
conda install pytorch torchvision cudatoolkit=[VERSION] -c pytorch # linux, choose the version for your system 

pip install setGPU tensorboardX
```

## Train

we provide the train scripts in the folder ```script```. Below is the example of training on noise sigma = 0.25
```

CUDA_VISIBLE_DEVICES=0 python code/train.py \
                        cifar10 \
                        cifar_resnet110 \
                        --lr 0.1 \
                        --epochs 150 \
                        --noise 0.25 \
                        --ds_hard \
                        --ds_epoch 60 \
                        --ds_threshold 0.5 \
                        --dataset_weight \
                        --dataset_weight_epoch 10 \
                        --adv \
                        --attacker pgd_radius_l2 \
                        --epsilon 0.25 \
                        --attack_steps 3 \
                        --id 1 \
```
To train under &sigma; = 0.25, 0.5, 1.0, you can run:
```
bash script/train_0.25.sh   # sigma=0.25
bash script/train_0.5.sh    # sigma=0.5
bash script/train_1.0.sh    # sigma=1.0
```

Checkpoints of our experiments can be find [here](https://mega.nz/folder/yVkkzBCS#L8DTNIjtA8w7iOx6cw7iNw). It contains three models trained under &sigma; = 0.25, 0.5, 1.0 respectively.

## Certification

All certification related code are from https://github.com/locuslab/smoothing.

We provide the scripts to run certifications in the folder ```scripts```. The follow is an example with &sigma;=0.25

```
CUDA_VISIBLE_DEVICES=0 python code/certify.py \
    cifar10 \
    logs/cifar10/0.25_3_pgd_radius_l2/ds_60_0.5/num_4/dataset_weight_10/noise_0.25/cifar_resnet110/1/checkpoint.pth.tar \
    0.25 \
    test/certify/cifar10/0.25_3_pgd_radius_l2/ds_60_0.5/num_4/dataset_weight_10/1/noise_0.25.tsv \
    --N=100000 \
    --skip=1 
```
To certify the model under &sigma; = 0.25, 0.5, 1.0, you can run
```
bash script/certify_0.25.sh    # sigma = 0.25
bash script/certify_0.5.sh     # sigma = 0.5
bash script/certify_1.0.sh     # sigma = 1.0
```