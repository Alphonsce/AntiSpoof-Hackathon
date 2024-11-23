# ckpt_path="checkpoints/Rawformer-2021-Train-no-aug/2021_train_ep_2_rawboost_algo_0_allow_aug_False.pth"
ckpt_path=checkpoints/Rawformer-2021-Train-no-aug/2021_train_ep_19_rawboost_algo_0_allow_aug_False.pth

CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture Rawformer \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file rawformer_19_ep_2021.csv