#!/bin/bash
#SBATCH --job-name=Weighted_Focal
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --partition=ai702
#SBATCH --reservation=ai702

nvidia-smi



for lr in  0.000005 
do
    for dataset in DR
    do
        for init in clip_full
        do
            for command in delete_incomplete launch
            do
                python -m domainbed.scripts.sweep $command\
                    --data_dir=/nfs/users/ext_group8/Dataset/224_data/ \
                    --output_dir=/nfs/users/ext_group8/Repos/BroaderDG/domainbed/Outputs/V9_CLIP_PROMPT_V2 \
                    --command_launcher multi_gpu\
                    --algorithms Clip_train_prompt_from_image \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 2 \
                    --hparams """{\"weight_init\":\"${init}\",\"backbone\":\"ClipBase\",\"lr\":${lr},\"batch_size\":20}"""\
                    --skip_confirmation
            done > /nfs/users/ext_group8/Repos/BroaderDG/Outs/V9_CLIP_PROMPT_V2.out
        done
    done
done

watch nvidia-smi