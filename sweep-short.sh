#!/bin/bash
#SBATCH --job-name=eye_
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5
#SBATCH --partition=default-short



# for dataset in DomainNet 
# do
#     for lr in 0.00005 
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_sanoojan.baliah/Sanoojan/DG/data \
#                 --output_dir=./domainbed/outputs_clip/Deitbase_related_ablations/ERM_Vit_with_clip_mix-0.6/${dataset}/lr-${lr}\
#                 --command_launcher multi_gpu_0_1\
#                 --algorithms ERM_Vit_with_clip_mix \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
#                 --skip_confirmation  
#         done > Outs/DeitBase-${dataset}-ERM_Vit_with_clip_mix.out
#     done
# done

nvidia-smi


# for lr in  0.000002 0.000001 0.000005
# do
#     for dataset in DR
#     do
#         for init in clip_full
#         do
#             for command in delete_incomplete launch
#             do
#                 python -m domainbed.scripts.sweep $command\
#                     --data_dir=/nfs/users/ext_group8/Dataset/224_data/ \
#                     --output_dir=./domainbed/Outputs/Eye_resnet-ERM_ViT_classifier_learning/${lr}\
#                     --command_launcher multi_gpu\
#                     --algorithms ERM_ViT_classifier_learning \
#                     --single_test_envs \
#                     --datasets ${dataset} \
#                     --n_hparams 1  \
#                     --n_trials 3 \
#                     --hparams """{\"lr\":${lr}}"""\
#                     --skip_confirmation  
#             done > Outs/ERM_ViT_classifier_learning.out
#         done
#     done
# done

for lr in 0.000005
do
    for dataset in DR
    do
        for init in clip_full
        do
            for command in delete_incomplete launch
            do
                python -m domainbed.scripts.sweep $command\
                    --data_dir=/nfs/users/ext_group8/Dataset/224_data/ \
                    --output_dir=./domainbed/Outputs/Clip_zero/${lr}\
                    --command_launcher multi_gpu\
                    --algorithms Clip_zero \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 1 \
                    --steps 3\
                    --hparams """{\"lr\":${lr}}"""\
                    --skip_confirmation  
            done > Outs/Clip_zero.out
        done
    done
done


