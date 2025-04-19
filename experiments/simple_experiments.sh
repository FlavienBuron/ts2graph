    . .venv/bin/activate

    # Get current date in YYMMDD format
    DATE=$(date +%y%m%d)
    LOGFILE="./experiments/results/${DATE}-simple-experiments.txt"

    echo "Running experiments on $DATE" >> "$LOGFILE"

    # Test the impact of hidden layer size

    # 1. Defaults 
    python -u main.py -d air -g zero 0 -i 1 -e 10 -hd 32 -o 16 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -i 1 -e 10 -hd 32 -o 16 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -i 1 -e 10 -hd 32 -o 16 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -i 1 -e 10 -hd 32 -o 16 -v 0 | tee -a "$LOGFILE"

