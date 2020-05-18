LOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 MC_COUNT_DISP=1000000 \
OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0  \
srun --mpi=pmi2 --job-name tt --partition=$1 -n1 --gres=gpu:1 --ntasks-per-node=1  \
python -u train_pytorch.py --gpu --batch_size 16 --base_lr 0.0005  #--resume 5 --test-only
