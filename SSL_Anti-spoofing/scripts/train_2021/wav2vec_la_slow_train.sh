CUDA_VISIBLE_DEVICES="5"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 24 \
    --comment wav2vec_la_slow_train \
    --algo 5 \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --train_year 2021 \
    --lr 2e-6