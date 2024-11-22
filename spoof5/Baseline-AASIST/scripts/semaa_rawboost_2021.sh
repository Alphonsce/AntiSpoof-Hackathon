CUDA_VISIBLE_DEVICES="4"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main.py \
    --config ./config/SEMAA_2021.conf \
    --comment rawboost \
    --use_rawboost \
    --algo 5