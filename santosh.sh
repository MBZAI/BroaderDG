#!/bin/bash
#SBATCH --job-name=Weighted_Focal
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --partition=ai702
#SBATCH --reservation=ai702

nvidia-smi


for lr in  0.00005 
do
    for dataset in DR
    do
        for init in clip_full
        do
            for command in delete_incomplete launch
            do
                CUDA_VISIBLE_DEVICES=7 python -m domainbed.scripts.sweep $command\
                    --data_dir=/nfs/users/ext_group8/Dataset/224_data/ \
                    --output_dir=./domainbed/Outputs/Eye_resnet-SANTOSH_V2 \
                    --command_launcher multi_gpu\
                    --algorithms ERM \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 1 \
                    --hparams """{\"lr\":${lr}}"""\
                    --skip_confirmation
            done > Outs/ERM_SANTOSH_V2.out
        done
    done
done