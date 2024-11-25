CUDA_VISIBLE_DEVICES="7"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main.py \
    --config ./config/SEMAA_2021_low_lr.conf \
    --comment low_lr_NO_AUG \
    --algo 0 \
    # --use_rawboost \
    # --algo 0