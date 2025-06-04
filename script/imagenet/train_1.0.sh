CUDA_VISIBLE_DEVICES=0 python code/train.py \
                        imagenet \
                        resnet50 \
                        --lr 0.1 \
                        --epochs 90 \
                        --noise 1.0 \
                        --num-noise-vec 2 \
                        --ds_hard \
                        --ds_epoch 0 \
                        --ds_threshold 0.1 \
                        --adv \
                        --attacker pgd_radius_l2 \
                        --epsilon 2.0 \
                        --attack_steps 1 \
                        --p_corr_file  p_correct/train/imagenet/-1_50.npy \
                        --id 1 \
