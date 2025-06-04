CUDA_VISIBLE_DEVICES=0 python code/certify.py \
    imagenet \
    logs/imagenet/0.5_2_pgd_radius_l2/ds_0_0.2/num_1/noise_0.25/resnet50/1/checkpoint.pth.tar \
    0.25 \
    test/certify/imagenet/0.5_2_pgd_radius_l2/ds_0_0.2/num_1/1/noise_0.25.tsv \
    --N=100000 \
    --skip=1 