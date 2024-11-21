CUDA_VISIBLE_DEVICES="4"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 16 \
    --comment xlsr_ssl \
    --algo 5