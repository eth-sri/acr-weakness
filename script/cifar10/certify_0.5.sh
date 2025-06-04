CUDA_VISIBLE_DEVICES=0 python code/certify.py \
    cifar10 \
    logs/cifar10/0.25_6_pgd_radius_l2/ds_70_0.4/num_4/dataset_weight_10/noise_0.5/cifar_resnet110/1/checkpoint.pth.tar \
    0.5 \
    test/certify/cifar10/0.25_6_pgd_radius_l2/ds_70_0.4/num_4/dataset_weight_10/1/noise_0.5.tsv \
    --N=100000 \
    --skip=1 