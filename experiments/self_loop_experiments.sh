#!/bin/bash

. .venv/bin/activate

EPOCHS=()
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=1
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

# Accept custom list of epochs from command line, or use defaults
if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=(5 10 15 20)
fi

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
    ["zero_0"]=0
    ["zero_1"]=1
    ["one_1"]=1
    ["one_0"]=0
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

        # Reset default self-loop
        SELF_LOOP=1
        BASE_G=$G

        # Check if technique is a variant of zero or one
        if [[ "$G" == zero_* ]]; then
            BASE_G="zero"
            SELF_LOOP=${G#zero_}
        elif [[ "$G" == one_* ]]; then
            BASE_G="one"
            SELF_LOOP=${G#one_}
        fi

        echo "Running: -g $BASE_G $V -e $E" | tee -a "$LOGFILE"
        TIMESTAMP=$(date +%y%m%d_%H%M%S)
        FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_ln${LAYER_NUMBER}_${BASE_G}_${V}_${E}.json"
        python -u main.py -d $DATASET -sp $FILENAME -g "$BASE_G" "$V" -e "$E" -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR $USE_TEMP -sl $SELF_LOOP -v 0 | tee -a "$LOGFILE"
    done
done.py -d air -g knn 50 -e 10 -hd 32 -sl 1 -ln 1 -v 0 | tee -a "$LOGFILE"    
