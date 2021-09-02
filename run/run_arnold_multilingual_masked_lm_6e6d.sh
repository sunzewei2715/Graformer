#!/bin/bash

export NCCL_IB_DISABLE=1

#            1          2        3
# script <dataset> <hdfs-dir> <suffix>

dataset=$1
hdfs_dir=$2
suffix=$3

IFS=' ' read -r -a extra_args <<< "${@: 3}"
echo "[logging] extra args: ${extra_args[@]}"

base_dir=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/sunzewei.v
dataset_path=${base_dir}/${dataset}
echo "[logging] dataset_path: ${dataset_path}"

#tensorboard_logdir=${ARNOLD_WORK_DIR}
# hdfs dfs -mkdir -p ${tensorboard_logdir}

local_root=/opt/tiger/fairseq/train
local_dataset_path=${local_root}/data
local_checkpoint_path=${local_root}/checkpoints
mkdir -p ${local_dataset_path}
mkdir -p ${local_checkpoint_path}

# download resource
hadoop fs -copyToLocal ${dataset_path}/* ${local_dataset_path}
#hadoop fs -copyToLocal ${dataset_path}/am ${dataset_path}/bg ${dataset_path}/bn ${dataset_path}/bs ${dataset_path}/dict.txt ${local_dataset_path}
echo "[logging] local_dataset_path: ${local_dataset_path}"
echo "[logging] local_checkpoint_path: ${local_checkpoint_path}"

if [ $ARNOLD_NUM != 1 ]
then
    CMD="python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_NUM --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --master_port=$METIS_WORKER_0_PORT"
else
    CMD="python3"
fi

echo "[logging] CMD: ${CMD}"
echo "[logging] Start Running..."

lang_list="am,bg,bn,bs,cs,de,el,en,es,et,fa,fi,fr,gu,hi,hr,hu,it,iu,ja,kk,km,kn,ky,lt,lv,mk,ml,mr,nl,or,pa,pl,ps,pt,ro,ro_kd,ru,so,sr,sw,ta,te,tr,uk,zh"

local_data=${local_dataset_path}
local_data="${local_dataset_path}/data-bin-0"
for i in {1..19} ; do
    local_data=${local_data}:${local_dataset_path}/data-bin-${i}
done

$CMD ../train.py --num-workers 8 \
    ${local_data} \
    --valid-subset none \
    --task new_multilingual_masked_lm \
    --langs ${lang_list} \
    --multilang-sampling-alpha 0.7 \
    --sample-break-mode eos --replace-mask-with-bos \
    --arch transformer_encoder_model_6l_16h_1024 \
    --encoder-learned-pos \
    --no-scale-embedding \
    --encoder-normalize-before --share-encoder-input-output-embed \
    --activation-fn gelu \
    --max-epoch 200 \
    --max-tokens 8000 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.005 --warmup-init-lr '1e-07' --min-loss-scale 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --update-freq 6  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.1 \
    --disable-validation \
    --save-interval 10 \
    --keep-last-epochs 20 \
    --save-dir ${local_checkpoint_path} \
    --fp16 \
    --ddp-backend=no_c10d $(echo ${extra_args[@]})
    #    --tensorboard-logdir "${tensorboard_logdir}/${signatures}" \

sleep $(($RANDOM/10))

checkpoint_path=${base_dir}/${hdfs_dir}
hadoop fs -test -e ${checkpoint_path}
if [ $? -eq 0 ] ; then
    timestamp=`date "+%Y-%m-%d-%H-%M-%S"`
    checkpoint_path=${checkpoint_path}_${timestamp}
    echo "[logging] !!! appointed checkpoint_path exists, changing to ${checkpoint_path}"
else
    echo "[logging] checkpoint_path: ${checkpoint_path}"
fi

hadoop fs -mkdir -p ${checkpoint_path}
hadoop fs -put -f ${local_checkpoint_path}/* ${checkpoint_path}
