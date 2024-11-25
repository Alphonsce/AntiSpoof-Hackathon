CUDA_VISIBLE_DEVICES="4"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 24 \
    --comment wav2vec_low_lr_DF_no_aug \
    --algo 0 \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --train_year 2021_DF \
    --lr 3e-6