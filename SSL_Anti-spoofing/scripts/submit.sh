CUDA_VISIBLE_DEVICES="7"

ckpt_path=models/model_LA_weighted_CCE_100_24_2e-06_wav2vec_codecs_rawboost_DF/ep_3_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture wav2vec \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file wav2vec_DF_rawboost_codec_ep_3.csv \
    --need_sigmoid \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \