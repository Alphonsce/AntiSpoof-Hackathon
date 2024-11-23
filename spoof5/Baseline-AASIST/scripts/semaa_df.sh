CUDA_VISIBLE_DEVICES="6"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main.py \
    --config ./config/SEMAA_DF_2021.conf \
    --comment semaa_DF_2021_NO_AUG \
    --algo 0
    # --use_rawboost \
    # --algo 