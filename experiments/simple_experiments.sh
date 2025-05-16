#!/bin/bash

. .venv/bin/activate

# Accept custom list of depth from command line, or use defaults
DEPTHS=("$@")
if [ ${#DEPTHS[@]} -eq 0 ]; then
    DEPTHS=(1 2 3)
fi

EPOCH=50
HIDDEN_DIM=32
SELF_LOOP=0
USE_MLP_OUTPUT=0
MLP_SIZE=32

DATE=$(date +%y%m%d)
LOGFILE="./experiments/results/${DATE}-simple-experiments.txt"

echo "Running experiments on $DATE" >> "$LOGFILE"

# List of grouping methods and their values
declare -A TECHNIQUES=(
    ["zero"]=0
    ["one"]=1
    ["loc"]=0.5
    ["knn"]=50
)

USE_MLP = ""
if [ "$USE_MLP_OUTPUT" -eq 1 ]; then
    USE_MLP="-mo -ms $MLP_SIZE"
fi

# Loop through epochs and groups
for D in "${DEPTHS[@]}"; do
    for T in "${!TECHNIQUES[@]}"; do
        V=${TECHNIQUES[$T]}
        echo "Running: -g $T $V -ln $D" | tee -a "$LOGFILE"
        python -u main.py -d air -g "$T" "$V" -e $EPOCH -hd $HIDDEN_DIM -ln $D $USE_MLP -v 0 | tee -a "$LOGFILE"
    done
done



