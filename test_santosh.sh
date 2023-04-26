#!/bin/bash
#SBATCH --job-name=Weighted_CE
#SBATCH --gres gpu:2
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
                CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.sweep $command\
                    --data_dir=/nfs/users/ext_group8/Dataset/224_data/ \
                    --output_dir=./domainbed/Outputs/Eye_resnet-ERM_weighted_CE_santosh_v19_biobert_with_new_prompts-lr${lr}\
                    --command_launcher multi_gpu\
                    --algorithms Clip_train \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 1 \
                    --hparams """{\"weight_init\":\"${init}\",\"backbone\":\"clipbase\",\"lr\":${lr}}"""\
                    --skip_confirmation
            done > Outs/ERM_weighted_CE_santosh_v19_second_exp.out
        done
    done
done