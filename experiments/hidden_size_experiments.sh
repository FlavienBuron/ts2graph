run() {
    . .venv/bin/activate

    # Get current date in YYMMDD format
    DATE=$(date +%y%m%d)
    LOGFILE="./experiments/${DATE}-hs-experiments.txt"

    echo "Running experiments on $DATE" >> "$LOGFILE"

    # Test the impact of hidden layer size

    # 1. Defaults 
    python -u main.py -d air -g zero 0 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"

    # 2. Size 8 
    python -u main.py -d air -g zero 0 -i 1 -e 10 -hd 8 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -i 1 -e 10 -hd 8 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -i 1 -e 10 -hd 8 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -i 1 -e 10 -hd 8 -o 16 | tee -a "$LOGFILE"

    # 3. Size 16
    python -u main.py -d air -g zero 0 -i 1 -e 10 -hd 16 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -i 1 -e 10 -hd 16 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -i 1 -e 10 -hd 16 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -i 1 -e 10 -hd 16 -o 16 | tee -a "$LOGFILE"

    # 4. Size 64
    python -u main.py -d air -g zero 0 -i 1 -e 10 -hd 64 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -i 1 -e 10 -hd 64 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -i 1 -e 10 -hd 64 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -i 1 -e 10 -hd 64 -o 16 | tee -a "$LOGFILE"
}
