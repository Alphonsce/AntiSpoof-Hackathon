CUDA_VISIBLE_DEVICES="5"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 24 \
    --comment wav2vec_codecs_BIG_data \
    --algo 0 \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --train_year 2021_DF_BIG \
    --lr 4e-6