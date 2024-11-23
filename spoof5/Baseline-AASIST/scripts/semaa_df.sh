CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main.py \
    --config ./config/SEMAA_DF_2021.conf \
    --comment semaa_DF_2021_rawboost_4 \
    --use_rawboost \
    --algo 4