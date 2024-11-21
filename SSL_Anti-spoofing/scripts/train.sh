CUDA_VISIBLE_DEVICES="6"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 24 \
    --comment dp_hubert_weighted_sum_unfreeze \
    --algo 5 \
    --ssl_backbone dp_hubert \
    --ssl_behaviour weighted-sum
    # --freeze_ssl 
    # --ssl_behaviour last-layer