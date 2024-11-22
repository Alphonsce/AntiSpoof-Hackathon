CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 24 \
    --comment 2021_dp_hubert_unfreeze \
    --algo 5 \
    --ssl_backbone dp_hubert \
    --ssl_behaviour last-layer \
    --train_year 2021