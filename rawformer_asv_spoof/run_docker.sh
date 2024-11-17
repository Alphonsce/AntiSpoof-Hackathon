DATA_PATH=/data/a.varlamov/asvspoof

docker run -it --rm \
    --network=host --shm-size=10g \
    --gpus "all" \
    -p 8888:8888 \
    -v $PWD:/app \
    -v $DATA_PATH:/dataset \
    ${name:+--name "$name"} \
    rawformer