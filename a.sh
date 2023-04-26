#!/bin/bash
#SBATCH --job-name=ckeck
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --partition=ai702
#SBATCH --reservation=ai702

nvidia-smi


source ~/.bashrc


for lr in  0.00005 
do
    for dataset in DR
    do
        for init in clip_full
        do
            for command in delete_incomplete launch
            do
                /nfs/users/ext_sanoojan.baliah/miniconda3/envs/dgbed4/bin/python -m domainbed.scripts.sweep $command\
                    --data_dir=/nfs/users/ext_group8/Dataset/224_data/ \
                    --output_dir=./domainbed/Outputs/Eye_resnet-ERM_check_class_imb-lr${lr}\
                    --command_launcher multi_gpu\
                    --algorithms ERM \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 1 \
                    --hparams """{\"lr\":${lr}}"""\
                    --skip_confirmation  
            done > Outs/ERM_check.out
        done
    done
done
