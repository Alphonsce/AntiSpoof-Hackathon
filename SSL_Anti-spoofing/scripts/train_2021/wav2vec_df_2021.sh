CUDA_VISIBLE_DEVICES="5"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 32 \
    --comment wav2vec_2021_DF \
    --algo 5 \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --train_year 2021_DF \
    --weight_decay 3e-5