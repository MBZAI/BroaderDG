python -m domainbed.scripts.sweep launch\
    --data_dir=/nfs/users/ext_group8/Dataset/224_data/ \
    --output_dir=./domainbed/Outputs/Eye_resnet-adam-check-rnd\
    --command_launcher multi_gpu\
    --algorithms ERM_focal_loss \
    --single_test_envs \
    --datasets DR \
    --n_hparams 1  \
    --n_trials 1 \
    --hparams """{\"lr\":-1}"""\
    --skip_confirmation


# srun -N 1 --gres=gpu:2 --time=01:00:00 --reservation=cv703 --partition=cv703 --pty bash

python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/ADAM_RND
python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/Eye_resnet-adam-check-focal-lr0.00005
python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/Eye_resnet-adam-check-lr0.00005
python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/Eye_resnet-adam-check-focal-lr0.00005


python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/V1_CLIP_BASELINE
python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/V2_CLIP_BIOBERT
python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/V3_CLIP_BIOBERT_ADVANCED

python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/V5_CLIP_BIOBERT_FOCAL

python -m domainbed.scripts.collect_results --input_dir domainbed/Outputs/V7_BIOMEDICAL_PROMPT
