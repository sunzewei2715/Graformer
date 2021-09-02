#!/bin/bash

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

# ted
lang_pairs="bg-en,bn-en,bs-en,cs-en,de-en,el-en,es-en,et-en,fa-en,fi-en,fr-en,hi-en,hr-en,hu-en,it-en,ja-en,kk-en,lt-en,mk-en,mr-en,nl-en,pl-en,pt-en,ro-en,ru-en,sr-en,ta-en,tr-en,uk-en,zh-en,en-bg,en-bn,en-bs,en-cs,en-de,en-el,en-es,en-et,en-fa,en-fi,en-fr,en-hi,en-hr,en-hu,en-it,en-ja,en-kk,en-lt,en-mk,en-mr,en-nl,en-pl,en-pt,en-ro,en-ru,en-sr,en-ta,en-tr,en-uk,en-zh"
lang_list="am,bg,bn,bs,cs,de,el,en,es,et,fa,fi,fr,gu,hi,hr,hu,it,iu,ja,kk,km,kn,ky,lt,lv,mk,ml,mr,nl,or,pa,pl,ps,pt,ro,ro_kd,ru,so,sr,sw,ta,te,tr,uk,zh"

$CMD ../train.py --num-workers 8 \
    ${local_dataset_path} \
    --task translation_multi_simple_epoch \
    --langs ${lang_list} --lang-pairs ${lang_pairs} \
    --sampling-method "temperature" --sampling-temperature 5 \
    --decoder-langtok --lang-tok-replacing-bos-eos \
    --arch transformer_vaswani_wmt_en_de_big \
    --encoder-layers 12 --decoder-layers 12 \
    --encoder-learned-pos --decoder-learned-pos \
    --no-scale-embedding \
    --encoder-normalize-before --decoder-normalize-before --share-decoder-input-output-embed \
    --activation-fn gelu \
    --max-update 100000000 \
    --max-tokens 4000 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.001 --warmup-init-lr '1e-07' --min-loss-scale 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --update-freq 5  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.1 \
    --no-epoch-checkpoints \
    --disable-validation \
    --save-interval-updates 200 \
    --keep-interval-updates 50 \
    --save-dir ${local_checkpoint_path} \
    --fp16 \
    --ddp-backend=no_c10d $(echo ${extra_args[@]})
    #    --tensorboard-logdir "${tensorboard_logdir}/${signatures}" \

checkpoint_path=${base_dir}/${hdfs_dir}
hadoop fs -test -e ${checkpoint_path}
if [ $? -eq 0 ] ; then
    timestamp=`date "+%Y-%m-%d-%H-%M-%S"`
    checkpoint_path=${checkpoint_path}_${timestamp}
    echo "[logging] !!! appointed checkpoint_path exists, changing to ${checkpoint_path}"
else
    echo "[logging] checkpoint_path: ${checkpoint_path}"
fi

hadoop fs -mkdir -p ${checkpoint_path}/checkpoints
hadoop fs -put -f ${local_checkpoint_path}/* ${checkpoint_path}/checkpoints
echo "[logging] Upload checkpoints finished"