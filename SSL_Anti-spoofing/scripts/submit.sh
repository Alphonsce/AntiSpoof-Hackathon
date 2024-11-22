ckpt_path="models/model_LA_weighted_CCE_100_24_1e-05_2021_wav2vec_unfreeze/last.pth"

CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture wav2vec \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file wav2vec_submit_sigmoid.csv \
    --need_sigmoid \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer