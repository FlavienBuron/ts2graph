#!/bin/bash

. .venv/bin/activate

EPOCHS=50
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
USE_TEMPORAL=0
MLP_SIZE=32
DATASET="airq_small"
NUM_NODES=36
FRACTION=0.05

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            EPOCHS="$2"
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
        --temporal)
            USE_TEMPORAL=1
            shift
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

if [[ "$DATASET" == "airq" ]]; then
    NUM_NODES=437
fi

if [[ -z "$LR" || "$LR" == "0" ]]; then
    if [[ "$LAYER_NUMBER" -eq 1 ]]; then
        LR=0.005
    else
        LR=0.0005
    fi
fi


DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/epochs/"
LOGFILE="${EXP_DIR}${DATE}-knn-experiments.txt"

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

USE_TEMP=""
if [ "$USE_TEMPORAL" -eq 1 ]; then
    USE_TEMP="-ut"
fi

# Loop through fixed graphs techniques
for G in "${!TECHNIQUES[@]}"; do
    V=${TECHNIQUES[$G]}
    echo "Running: -g $G $V -e $EPOCHS" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_ln${LAYER_NUMBER}_${G}_${V}_${EPOCHS}.json"
    python -u main.py -d $DATASET -sp $FILENAME -g "$G" "$V" -e "$EPOCHS" -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR $USE_TEMP -v 0 | tee -a "$LOGFILE"
done


# Sweep knn values from 1 to KNN_MAX
for LOC in $(seq 0.0 $FRACTION 1.0); do
    printf -v LOC_FMT "%.2f" "$LOC"
    echo "Running: -g loc $LOC_FMT -e $EPOCHS" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_ln${LAYER_NUMBER}_loc_${LOC_FMT}_${EPOCHS}.json"
    python -u main.py -d $DATASET -sp $FILENAME -g loc $LOC_FMT -e $EPOCHS \
           -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR $USE_TEMP -v 0 | tee -a "$LOGFILE"
done
