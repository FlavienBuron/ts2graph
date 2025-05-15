#!/bin/bash

. .venv/bin/activate

# Accept custom list of epochs from command line, or use defaults
EPOCHS=("$@")
if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=(5 10 15 20)
fi

DATE=$(date +%y%m%d)
LOGFILE="./experiments/results/${DATE}-e-experiments.txt"

echo "Running experiments on $DATE" >> "$LOGFILE"

# List of grouping methods and their values
declare -A TECHNIQUES=(
    ["zero"]=0
    ["one"]=1
    ["loc"]=0.5
    ["knn"]=50
)

# Loop through epochs and groups
for E in "${EPOCHS[@]}"; do
    for G in "${!TECHNIQUES[@]}"; do
        V=${TECHNIQUES[$G]}
        echo "Running: -g $G $V -e $E" | tee -a "$LOGFILE"
        python -u main.py -d air -g "$G" "$V" -e "$E" -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    done
done

