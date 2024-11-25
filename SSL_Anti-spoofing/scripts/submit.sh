CUDA_VISIBLE_DEVICES="6"

ckpt_path=models/model_LA_weighted_CCE_100_24_3e-06_wav2vec_semaa_la/ep_5_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture wav2vec \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file wav2vec_SEMAA_rawboost_low_lr_5_epochs.csv \
    --need_sigmoid \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --use_semaa

ckpt_path=models/model_LA_weighted_CCE_100_24_2e-06_wav2vec_la_slow_train/ep_5_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture wav2vec \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file wav2vec_rawboost_low_lr_5_epochs.csv \
    --need_sigmoid \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer