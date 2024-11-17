#!/bin/bash


ENV_PATH=".env"	ENV_PATH=".env"
export $(grep -v '^#' "$ENV_PATH" | xargs)	export $(cat .env | grep -v '#' | awk '/=/ {print $1}')


wandb login $WANDB_API_KEY	wandb login $WANDB_API_KEY