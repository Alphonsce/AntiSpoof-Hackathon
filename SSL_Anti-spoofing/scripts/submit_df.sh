ckpt_path=models/model_LA_weighted_CCE_100_32_1e-05_wav2vec_2021_DF/ep_4_ckpt.pth

CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture wav2vec \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file wav2vec_4_epochs_DF.csv \
    --need_sigmoid \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-laye4r