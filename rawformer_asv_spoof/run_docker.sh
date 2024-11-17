DATA_PATH=/data/a.varlamov/ASVspoof2021_LA_eval

docker run -it --rm \
    --network=host --shm-size=10g \
    --gpus "all" \
    -p 8888:8888 \
    -v $PWD:/app \
    -v $DATA_PATH:/app/data \
    ${name:+--name "$name"} \
    matcha