#!/bin/bash

#            1      2
# script <script> <args>

cd $(dirname $0)
sudo pip install --editable ../

cd /opt/tiger/fairseq/run
echo "current dir: `pwd`"
echo "[logging] args: $@"

echo "[logging] running shell script: $1"
bash $1 ${@: 2}
echo "[logging] shell script finished: $1"
