#!/bin/bash

. .venv/bin/activate

EPOCHS=35
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
STGI_MODE='s'
MLP_SIZE=32
DATASET="airq_small"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --self-loop)
            SELF_LOOP=1
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --layers)
            LAYER_NUMBER="$2"
            shift 2
            ;;
        --mode)
            STGI_MODE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$LR" || "$LR" == "0" ]]; then
    if [[ "$LAYER_NUMBER" -eq 1 ]]; then
        LR=0.005
    else
        LR=0.0005
    fi
fi


DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/self-loop/"
LOGFILE="${EXP_DIR}${DATE}-sl-experiments.txt"

mkdir -p "$EXP_DIR"

echo "Running experiments on $DATE" >> "$LOGFILE"

KNN_VAL=50
[ "$DATASET" == "airq_small" ] && KNN_VAL=3

declare -A TECHNIQUES=(
    ["zero"]=0
    ["one"]=1
    ["loc"]=0.5
    ["knn"]=$KNN_VAL
)

# Loop through epochs and groups
for SELF_LOOP in 0 1; do
    for G in "${!TECHNIQUES[@]}"; do
        V=${TECHNIQUES[$G]}
        echo "Running: -g $G $V -e $EPOCHS" | tee -a "$LOGFILE"
        TIMESTAMP=$(date +%y%m%d_%H%M%S)
        FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_ln${LAYER_NUMBER}_${G}_${V}_sl${SELF_LOOP}_${EPOCHS}.json"
        python -u main.py -d $DATASET -sp $FILENAME -g "$G" "$V" -e $EPOCHS -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP -v 0 | tee -a "$LOGFILE"
    done
done
