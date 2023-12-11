echo "running time_matmul_no_sync 5 times"
time for i in {1..5}; do python time_matmul_no_synchronize.py; done
