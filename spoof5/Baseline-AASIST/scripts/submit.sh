# ckpt_path="exp_result/SEMAA_2021_ep100_bs24_rawboost/weights/epoch_1_0.001.pth"
# ckpt_path=exp_result/SEMAA_ep100_bs24_rawboost/weights/epoch_20_0.018.pth

ckpt_path=exp_result/SEMAA_2021_ep100_bs24_NO_AUG/weights/epoch_20_0.001.pth

CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture SEMAA \
    --ckpt_path $ckpt_path \
    --config_path ./config/SEMAA_2021.conf \
    --device cuda \
    --output_file semaa2021_NO_AUG_20_ep.csv \
    # --need_sigmoid