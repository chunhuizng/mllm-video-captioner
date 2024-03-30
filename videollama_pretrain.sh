#!/usr/bin/env bash

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0

torchrun \
--nnodes=$ARNOLD_WORKER_NUM \
--node_rank=$ARNOLD_ID \
--nproc_per_node=$ARNOLD_WORKER_GPU \
--master_addr=$METIS_WORKER_0_HOST \
--master_port=$port \
train.py --cfg-path lavis/projects/videollama/train/pretrain.yaml $@