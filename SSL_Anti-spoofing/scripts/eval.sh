CUDA_VISIBLE_DEVICES="4"

MODEL_PATH=models/model_LA_weighted_CCE_100_16_0.0001_xlsr_ssl/last.pth

EVAL_OUTPUT=eval_output/eval_xlsr.txt

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main_SSL_LA.py \
    --batch_size 16 \
    --eval \
    --eval_output $EVAL_OUTPUT \
    --model_path $MODEL_PATH

python evaluate_2021_LA.py $EVAL_OUTPUT /data/a.varlamov/asvspoof/ASVspoof2021_LA_eval/keys/LA/ eval