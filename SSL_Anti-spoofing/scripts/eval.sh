# CUDA_VISIBLE_DEVICES="4"

# MODEL_PATH=models/model_LA_weighted_CCE_100_16_0.0001_xlsr_ssl/last.pth

# EVAL_OUTPUT=eval_output/eval_xlsr.txt

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
#     --batch_size 16 \
#     --eval \
#     --eval_output $EVAL_OUTPUT \
#     --model_path $MODEL_PATH

# python evaluate_2021_LA.py $EVAL_OUTPUT /data/a.varlamov/asvspoof/ASVspoof2021_LA_eval/keys/LA/ eval

CUDA_VISIBLE_DEVICES="7"

# COMMENT=dp_hubert_last_layer_freeze

# MODEL_PATH=models/model_LA_weighted_CCE_100_24_0.0001_${COMMENT}/last.pth

# EVAL_OUTPUT=eval_output/eval_${COMMENT}.txt

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
#     --batch_size 16 \
#     --eval \
#     --eval_output $EVAL_OUTPUT \
#     --model_path $MODEL_PATH

# python evaluate_2021_LA.py $EVAL_OUTPUT /data/a.varlamov/asvspoof/ASVspoof2021_LA_eval/keys/LA/ eval

# -------------------

# COMMENTS=("dp_hubert_last_layer_unfreeze" "dp_hubert_last_layer_freeze")

# for COMMENT in "${COMMENTS[@]}"; do

#     MODEL_PATH=models/model_LA_weighted_CCE_100_24_0.0001_${COMMENT}/last.pth

#     EVAL_OUTPUT=eval_output/eval_${COMMENT}.txt

#     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
#         --batch_size 32 \
#         --eval \
#         --eval_output $EVAL_OUTPUT \
#         --model_path $MODEL_PATH \
#         --ssl_backbone dp_hubert \
#         --ssl_behaviour last-layer

#     python evaluate_2021_LA.py $EVAL_OUTPUT /data/a.varlamov/asvspoof/ASVspoof2021_LA_eval/keys/LA/ eval
# done

# ================== Weighted-sum:

# COMMENT=dp_hubert_weighted_sum_unfreeze
# MODEL_PATH=models/model_LA_weighted_CCE_100_24_0.0001_${COMMENT}/last.pth

# EVAL_OUTPUT=eval_output/eval_${COMMENT}.txt

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
#     --batch_size 32 \
#     --eval \
#     --eval_output $EVAL_OUTPUT \
#     --model_path $MODEL_PATH \
#     --ssl_backbone dp_hubert \
#     --ssl_behaviour weighted-sum

# python evaluate_2021_LA.py $EVAL_OUTPUT /data/a.varlamov/asvspoof/ASVspoof2021_LA_eval/keys/LA/ eval

# ================

COMMENT="dp_hubert_last_layer_unfreeze"

MODEL_PATH=models/model_LA_weighted_CCE_100_24_0.0001_${COMMENT}/last.pth

EVAL_OUTPUT=eval_output/eval_${COMMENT}.txt

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 32 \
    --eval \
    --eval_output $EVAL_OUTPUT \
    --model_path $MODEL_PATH \
    --ssl_backbone dp_hubert \
    --ssl_behaviour last-layer

python evaluate_2021_LA.py $EVAL_OUTPUT /data/a.varlamov/asvspoof/ASVspoof2021_LA_eval/keys/LA/ eval