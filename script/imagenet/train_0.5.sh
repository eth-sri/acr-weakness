CUDA_VISIBLE_DEVICES=0 python code/train.py \
                        imagenet \
                        resnet50 \
                        --lr 0.1 \
                        --epochs 90 \
                        --noise 0.5 \
                        --num-noise-vec 1 \
                        --ds_hard \
                        --ds_epoch 0 \
                        --ds_threshold 0.2 \
                        --adv \
                        --attacker pgd_radius_l2 \
                        --epsilon 1.0 \
                        --attack_steps 2 \
                        --p_corr_file  p_correct/train/imagenet/-1_50.npy \
                        --id 1 \