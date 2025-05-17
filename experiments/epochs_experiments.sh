#!/bin/bash

. .venv/bin/activate

# Accept custom list of epochs from command line, or use defaults
EPOCHS=("$@")
if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=(5 10 15 20)
fi

HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
USE_MLP_OUTPUT=0
MLP_SIZE=32
DATASET="airq_small"

DATE=$(date +%y%m%d)
LOGFILE="./experiments/results/${DATE}-e-experiments.txt"

echo "Running experiments on $DATE" >> "$LOGFILE"

KNN_VAL=50
[ "$DATASET" == "airq_small" ] && KNN_VAL=3

declare -A TECHNIQUES=(
    ["zero"]=0
    ["one"]=1
    ["loc"]=0.5
    ["knn"]=$KNN_VAL
)

USE_MLP=""
if [ "$USE_MLP_OUTPUT" -eq 1 ]; then
    USE_MLP="-mo -ms $MLP_SIZE"
fi

# Loop through epochs and groups
for E in "${EPOCHS[@]}"; do
    for G in "${!TECHNIQUES[@]}"; do
        V=${TECHNIQUES[$G]}
        echo "Running: -g $G $V -e $E" | tee -a "$LOGFILE"
        python -u main.py -d $DATASET -g "$G" "$V" -e "$E" -hd $HIDDEN_DIM -ln $LAYER_NUMBER $USE_MLP -v 0 | tee -a "$LOGFILE"
    done
done

