. .venv/bin/activate

# Get current date in YYMMDD format
DATE=$(date +%y%m%d)
LOGFILE="${DATE}-experiments.txt"

echo "Running experiments on $DATE" >> "$LOGFILE"

# Test the impact of hidden layer size

# 1. Defaults 
python main.py -d air -g zero 0 -i 1 -e 10 -hd 32 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g one 1 -i 1 -e 10 -hd 32 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g loc 0.5 -i 1 -e 10 -hd 32 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g knn 100 -i 1 -e 10 -hd 32 -o 16 >> "$LOGFILE" 2>&1

# 2. Size 8 
python main.py -d air -g zero 0 -i 1 -e 10 -hd 8 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g one 1 -i 1 -e 10 -hd 8 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g loc 0.5 -i 1 -e 10 -hd 8 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g knn 100 -i 1 -e 10 -hd 8 -o 16 >> "$LOGFILE" 2>&1

# 3. Size 16
python main.py -d air -g zero 0 -i 1 -e 10 -hd 16 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g one 1 -i 1 -e 10 -hd 16 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g loc 0.5 -i 1 -e 10 -hd 16 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g knn 100 -i 1 -e 10 -hd 16 -o 16 >> "$LOGFILE" 2>&1

# 4. Size 64
python main.py -d air -g zero 0 -i 1 -e 10 -hd 64 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g one 1 -i 1 -e 10 -hd 64 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g loc 0.5 -i 1 -e 10 -hd 64 -o 16 >> "$LOGFILE" 2>&1
python main.py -d air -g knn 100 -i 1 -e 10 -hd 64 -o 16 >> "$LOGFILE" 2>&1

