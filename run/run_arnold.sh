#!/bin/bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

cd $(dirname $0)
sudo pip install --editable ../ -i https://pypi.doubanio.com/simple

cd /opt/tiger/fairseq/run
echo "current dir: `pwd`"
echo "[logging] args: $@"

hadoop fs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/sunzewei.v/pretrain/scripts/$1

echo "[logging] running shell script: $1"
bash $1 ${@: 2}
echo "[logging] shell script finished: $1"
