CUDA_VISIBLE_DEVICES="5"

ckpt_path=exp_result/SEMAA_2021_ep100_bs24_NO_AUG/weights/epoch_50_0.000.pth

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture SEMAA \
    --ckpt_path $ckpt_path \
    --config_path ./config/SEMAA_2021.conf \
    --device cuda \
    --output_file semaa_NO_AUG_50_ep.csv \
    --need_sigmoid

ckpt_path=exp_result/SEMAA_2021_ep100_bs24_rawboost/weights/epoch_18_0.001.pth

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture SEMAA \
    --ckpt_path $ckpt_path \
    --config_path ./config/SEMAA_2021.conf \
    --device cuda \
    --output_file semaa_rawboost_18_ep.csv \
    --need_sigmoid

ckpt_path=exp_result/SEMAA_DF_2021_ep100_bs24_semaa_DF_2021_NO_AUG/weights/epoch_5_0.055.pth

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture SEMAA \
    --ckpt_path $ckpt_path \
    --config_path ./config/SEMAA_2021.conf \
    --device cuda \
    --output_file semaa_df_no_aug_5_ep.csv \
    --need_sigmoid