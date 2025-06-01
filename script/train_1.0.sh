CUDA_VISIBLE_DEVICES=0 python code/train.py \
                        cifar10 \
                        cifar_resnet110 \
                        --lr 0.1 \
                        --epochs 150 \
                        --noise 1.0 \
                        --ds_hard \
                        --ds_epoch 60 \
                        --ds_threshold 0.4 \
                        --dataset_weight \
                        --dataset_weight_epoch 10 \
                        --adv \
                        --attacker pgd_radius_l2 \
                        --epsilon 0.5 \
                        --attack_steps 4 \
                        --id 1 \
