prefix = 'srun --mpi=pmi2 --job-name tt --partition=VA -n1 --gres=gpu:1 --ntasks-per-node=1  -w SH-IDC1-10-5-37-51 '
import sys
import os
suffix = ' '.join(sys.argv[1:])
cmd = prefix + suffix
os.system(cmd)
