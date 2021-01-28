#!/bin/bash

echo "[logging] args: $@"

hadoop fs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/sunzewei.v/pretrain/scripts/$1

echo "[logging] running shell script: $1"
bash $1 ${@: 2}
echo "[logging] shell script finished: $1"
