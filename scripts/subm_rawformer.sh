ckpt_path="checkpoints/Rawformer-ACN-Aug/ep_3_rawboost_algo_0_allow_aug_True.pth"

CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture Rawformer \
    --ckpt_path $ckpt_path \
    --device cuda