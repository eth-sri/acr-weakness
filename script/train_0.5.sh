CUDA_VISIBLE_DEVICES=0 python code/train.py \
                        cifar10 \
                        cifar_resnet110 \
                        --lr 0.1 \
                        --epochs 150 \
                        --noise 0.5 \
                        --ds_hard \
                        --ds_epoch 70 \
                        --ds_threshold 0.4 \
                        --dataset_weight \
                        --dataset_weight_epoch 10 \
                        --adv \
                        --attacker pgd_radius_l2 \
                        --epsilon 0.25 \
                        --attack_steps 6 \
                        --id 1 \