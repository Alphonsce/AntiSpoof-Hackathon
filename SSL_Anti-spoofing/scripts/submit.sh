ckpt_path="exp_result/SEMAA_2021_ep100_bs24_rawboost/weights/epoch_1_0.001.pth"

CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture SEMAA \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file wav2vec_submit_sigmoid.csv \
    --need_sigmoid