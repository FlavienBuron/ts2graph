    . .venv/bin/activate

    # Get current date in YYMMDD format
    DATE=$(date +%y%m%d)
    LOGFILE="./experiments/results/${DATE}-it-experiments.txt"

    echo "Running experiments on $DATE" >> "$LOGFILE"

    # Test the impact of iteration number

    # 1. Defaults it=1
    python -u main.py -d air -g zero 0 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"

    # 2. it=2
    python -u main.py -d air -g zero 0 -e 10 -it 2 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -it 2 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -it 2 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -e 10 -it 2 -ln 1 -v 0 | tee -a "$LOGFILE"

    # 3. it=3
    python -u main.py -d air -g zero 0 -e 10 -it 3 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -it 3 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -it 3 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -e 10 -it 3 -ln 1 -v 0 | tee -a "$LOGFILE"

    # 4. it=4
    python -u main.py -d air -g zero 0 -e 10 -it 4 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -it 4 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -it 4 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -e 10 -it 4 -ln 1 -v 0 | tee -a "$LOGFILE"

    # 5. it=5
    python -u main.py -d air -g zero 0 -e 10 -it 4 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -it 4 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -it 4 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -e 10 -it 4 -ln 1 -v 0 | tee -a "$LOGFILE"
