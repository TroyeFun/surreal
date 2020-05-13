import sys
import os
prefix = 'srun --mpi=pmi2 --job-name tt --partition=VA -n1 --gres=gpu:1 --ntasks-per-node=1  -w SH-IDC1-10-5-37-51 '
port_prefix = sys.argv[1]

cmd_pat = prefix + 'lsof -i:{}{:02d}'
cmd = cmd_pat.format(port_prefix, 1) + ' > log.txt'
print(cmd)
os.system(cmd)
for port_suffix in range(2, 12):
    cmd = cmd_pat.format(port_prefix, port_suffix) + ' >> log.txt'
    print(cmd)
    os.system(cmd)

flog = open('log.txt', 'r')
jobs = []
for line in flog.readlines():
    if 'lixiaojie' in line:
        job = line.split(' ')[1]
        jobs.append(job)
        cmd = prefix + 'kill ' + job
        os.system(cmd)
print(jobs)
