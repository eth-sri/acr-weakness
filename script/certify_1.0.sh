CUDA_VISIBLE_DEVICES=0 python code/certify.py \
    cifar10 \
    logs/cifar10/0.5_4_pgd_radius_l2/ds_60_0.4/num_4/dataset_weight_10/noise_1.0/cifar_resnet110/1/checkpoint.pth.tar \
    1.0 \
    test/certify/cifar10/0.5_4_pgd_radius_l2/ds_60_0.4/num_4/dataset_weight_10/1/noise_1.0.tsv \
    --N=100000 \
    --skip=1 
