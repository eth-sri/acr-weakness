CUDA_VISIBLE_DEVICES=0 python code/certify.py \
    cifar10 \
    logs/cifar10/0.25_3_pgd_radius_l2/ds_60_0.5/num_4/dataset_weight_10/noise_0.25/cifar_resnet110/1/checkpoint.pth.tar \
    0.25 \
    test/certify/cifar10/0.25_3_pgd_radius_l2/ds_60_0.5/num_4/dataset_weight_10/1/noise_0.25.tsv \
    --N=100000 \
    --skip=1 