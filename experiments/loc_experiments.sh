    . .venv/bin/activate

    # Get current date in YYMMDD format
    DATE=$(date +%y%m%d)
    LOGFILE="./experiments/results/${DATE}-loc-experiments.txt"

    echo "Running experiments on $DATE" >> "$LOGFILE"

    # Test the impact of the loc argument size

    python -u main.py -d air -g loc 0.0 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.1 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.2 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.3 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.4 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.6 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.7 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.8 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.9 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 1.0 -i 1 -e 10 -hd 32 -o 16 | tee -a "$LOGFILE"
