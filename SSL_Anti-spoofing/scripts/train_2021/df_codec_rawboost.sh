CUDA_VISIBLE_DEVICES="4"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 24 \
    --comment wav2vec_codecs_rawboost_DF \
    --algo 5 \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --train_year 2021_DF \
    --lr 2e-06