CUDA_VISIBLE_DEVICES="5"

ckpt_path=models/model_LA_weighted_CCE_100_24_4e-06_wav2vec_NO_AUG_la/ep_7_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python submit.py \
    --architecture wav2vec \
    --ckpt_path $ckpt_path \
    --device cuda \
    --output_file wav2vec_no_aug_low_lr_7_epochs.csv \
    --need_sigmoid \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \