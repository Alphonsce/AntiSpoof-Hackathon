CUDA_VISIBLE_DEVICES="4"
EVAL_OUTPUT=pre_computed_scores/LA/Scores_LA.txt

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
#     --batch_size 32 \
#     --eval \
#     --eval_output $EVAL_OUTPUT

python evaluate_2021_LA.py $EVAL_OUTPUT /data/a.varlamov/asvspoof/ASVspoof2021_LA_eval/keys/LA/ eval