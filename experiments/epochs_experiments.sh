#!/bin/bash

. .venv/bin/activate

EPOCHS=()
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
USE_TEMPORAL=0
MLP_SIZE=32
DATASET="airq_small"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            IFS=',' read -r -a EPOCHS <<< "$2"
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
        --mlp)
            USE_TEMPORAL=1
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Accept custom list of epochs from command line, or use defaults
if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=(5 10 15 20)
fi


DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/epochs/"
LOGFILE="${EXP_DIR}${DATE}-e-experiments.txt"

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

# Loop through epochs and groups
for E in "${EPOCHS[@]}"; do
    for G in "${!TECHNIQUES[@]}"; do
        V=${TECHNIQUES[$G]}
        echo "Running: -g $G $V -e $E" | tee -a "$LOGFILE"
        TIMESTAMP=$(date +%y%m%d_%H%M%S)
        FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_${G}_${V}_${E}.json"
        python -u main.py -d $DATASET -sp $FILENAME -g "$G" "$V" -e "$E" -hd $HIDDEN_DIM -ln $LAYER_NUMBER $USE_TEMP -v 0 | tee -a "$LOGFILE"
    done
done

