CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 24 \
    --comment wav2vec_freeze \
    --algo 5 \
    --ssl_backbone wav2vec \
    --ssl_behaviour last-layer \
    --freeze_ssl 
    # --ssl_behaviour last-layer