CUDA_VISIBLE_DEVICES="4"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_dp_hubert_LA.py \
    --batch_size 16 \
    --comment testing \
    --algo 5