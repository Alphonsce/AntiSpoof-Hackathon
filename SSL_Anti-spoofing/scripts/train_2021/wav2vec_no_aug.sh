CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 24 \
    --comment wav2vec_NO_AUG_la \
    --algo 0 \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --train_year 2021 \
    --lr 4e-6 \