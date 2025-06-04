CUDA_VISIBLE_DEVICES=0 python code/certify.py \
    imagenet \
    logs/imagenet/2.0_1_pgd_radius_l2/ds_0_0.1/num_2/noise_1.0/resnet50/1/checkpoint.pth.tar \
    1.0 \
    test/certify/imagenet/2.0_1_pgd_radius_l2/ds_0_0.1/num_2/1/noise_1.0.tsv \
    --N=100000 \
    --skip=1 