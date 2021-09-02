#!/bin/bash

#            1          2        3          4
# script <dataset> <hdfs-dir> <ckpt-id> <suffix>

dataset=$1
hdfs_dir=$2
ckpt_id=$3
suffix=$4

IFS=' ' read -r -a extra_args <<< "${@: 4}"
echo "[logging] extra args: ${extra_args[@]}"

base_dir=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/sunzewei.v
dataset_path=${base_dir}/${dataset}
echo "[logging] dataset_path: ${dataset_path}"

checkpoint_path=${base_dir}/${hdfs_dir}
hadoop fs -test -e ${checkpoint_path}
if [ $? -eq 0 ] ; then
    echo "[logging] checkpoint_path: ${checkpoint_path}"
else
    echo "[logging] ${checkpoint_path} NOT FOUND"
    exit 1
fi

local_root=/opt/tiger/fairseq/train
local_dataset_path=${local_root}/data
local_checkpoint_path=${local_root}/checkpoints
mkdir -p ${local_dataset_path}
mkdir -p ${local_checkpoint_path}

echo "[logging] local_dataset_path: ${local_dataset_path}"
echo "[logging] local_checkpoint_path: ${local_checkpoint_path}"

echo "[logging] Downloading datasets"
hadoop fs -get ${dataset_path}/test* ${dataset_path}/dict*  ${local_dataset_path}

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
# test
lang_list="am,bg,bn,bs,cs,de,el,en,es,et,fa,fi,fr,gu,hi,hr,hu,it,iu,ja,kk,km,kn,ky,lt,lv,mk,ml,mr,nl,or,pa,pl,ps,pt,ro,ro_kd,ru,so,sr,sw,ta,te,tr,uk,zh"

cp -r ../examples/m2m_100 ./
cd m2m_100
bash install_dependecies.sh
cd ..

post_process="sentencepiece"

while true; do
    timestamp=`date "+%Y-%m-%d-%H-%M-%S"`
    echo "[logging] --------------------------- Current Time: ${timestamp} --------------------------- "
    local_hyp_path=${local_checkpoint_path}/hyps_${ckpt_id}
    mkdir -p ${local_hyp_path}
    echo "[logging] Downloading checkpoints"
    hadoop fs -get ${checkpoint_path}/checkpoints/${ckpt_id} ${local_checkpoint_path}
    echo "[logging] Start inference for the last checkpoint"
    IFS=',' read -ra lang_pair_list <<< ${lang_pairs}
    for lang_pair in ${lang_pair_list[@]}; do
        IFS='-' read -ra langs <<< ${lang_pair}
        source_lang=${langs[0]}
        target_lang=${langs[1]}
        echo "[logging] --------- ${source_lang}2${target_lang} --------- "
        now=`date "+%Y-%m-%d %H:%M:%S"`
        echo "[logging] Start Time: ${now}"

        $CMD ../fairseq_cli/generate.py \
            ${local_dataset_path} \
            --gen-subset test \
            --task translation_multi_simple_epoch \
            --decoder-langtok --lang-tok-replacing-bos-eos \
            --path ${local_checkpoint_path}/${ckpt_id} \
            --langs ${lang_list} \
            --lang-pairs ${lang_pairs} \
            --source-lang $source_lang \
            --target-lang $target_lang \
            --beam 4 --lenpen 0.6  \
            --post-process ${post_process} \
            --sacrebleu \
            --batch-size 2048 \
            --max-tokens 8192 \
            --model-overrides "{'inference':'True'}" > ${source_lang}2${target_lang}.log

        cat ${source_lang}2${target_lang}.log | grep -P "^H" | cut -f 3- > ${source_lang}2${target_lang}.hyp
        cat ${source_lang}2${target_lang}.log | grep -P "^T" | cut -f 2- > ${source_lang}2${target_lang}.ref

        echo "[logging] sacrebleu:"
        cat ${source_lang}2${target_lang}.hyp | sacrebleu ${source_lang}2${target_lang}.ref -l ${source_lang}-${target_lang}

        cd m2m_100
        cat ../${source_lang}2${target_lang}.hyp | sh tok.sh ${target_lang} > ../${source_lang}2${target_lang}.hyp.tok
        cat ../${source_lang}2${target_lang}.ref | sh tok.sh ${target_lang} > ../${source_lang}2${target_lang}.ref.tok
        cd ..

        echo "[logging] normalized tokenized bleu:"
        cat ${source_lang}2${target_lang}.hyp.tok | sacrebleu ${source_lang}2${target_lang}.ref.tok -l ${source_lang}-${target_lang} -tok none

        mv ${source_lang}2${target_lang}.*  ${local_hyp_path}

        now=`date "+%Y-%m-%d %H:%M:%S"`
        echo "[logging] End Time:   ${now}"
        echo "[logging] ------------------------- "
    done

    mv ${local_checkpoint_path}/checkpoint_last.pt ${local_checkpoint_path}/checkpoint_last_${timestamp}.pt

    hadoop fs -put -f ${local_hyp_path} ${local_checkpoint_path}/${ckpt_id} ${checkpoint_path}

    echo "[logging] Finish"
    exit 1
done
