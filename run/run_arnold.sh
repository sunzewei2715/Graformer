#!/bin/bash

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

cd $(dirname $0)
sudo pip install --editable ../ -i https://bytedpypi.byted.org/simple

cd /opt/tiger/fairseq/run
echo "current dir: `pwd`"
echo "[logging] args: $@"

hadoop fs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/sunzewei.v/pretrain/scripts/$1

echo "[logging] running shell script: $1"
bash $1 ${@: 2}
echo "[logging] shell script finished: $1"
