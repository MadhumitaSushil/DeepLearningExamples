#! /bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

train_batch_size=${1:-40}
learning_rate=${2:-"9.625e-5"}
cased=${3:-true}
precision=${4:-"fp32"}
use_xla=${5:-"true"}
num_gpu=${6:-2}
warmup_steps=${7:-"1953"}
train_steps=${8:-19531}
num_accumulation_steps=${9:-16}
save_checkpoint_steps=${10:-5000}
eval_batch_size=${11:-80}

use_fp16=""
if [ "$precision" = "fp16" ] ; then
        echo "fp16 activated!"
        use_fp16="--use_fp16"
fi

if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--use_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi

if [ "$cased" = "true" ] ; then
    DO_LOWER_CASE=0
    CASING_DIR_PREFIX="cased"
else
    DO_LOWER_CASE=1
    CASING_DIR_PREFIX="uncased"
fi

BERT_CONFIG=/workspace/bert/data/download/google_pretrained_weights/${CASING_DIR_PREFIX}_biobert_v1.1_pubmed/bert_config.json
RESULTS_DIR=/results
CHECKPOINTS_DIR=${RESULTS_DIR}/wikimed_phase_1
mkdir -p ${CHECKPOINTS_DIR}

INIT_CHECKPOINT=/workspace/bert/data/download/google_pretrained_weights/${CASING_DIR_PREFIX}_biobert_v1.1_pubmed/bert_model.ckpt

INPUT_FILES_DIR="/workspace/bert/data/tfrecord/lower_case_${DO_LOWER_CASE}_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_57540_test_split_10/wikimed/training"
EVAL_FILES_DIR="/workspace/bert/data/tfrecord/lower_case_${DO_LOWER_CASE}_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_57540_test_split_10/wikimed/test"

if [ $num_gpu -gt 1 ] ; then
    mpi_command="mpirun -np $num_gpu -H localhost:$num_gpu \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib"
    use_hvd="--horovod"
else
    mpi_command=""
    use_hvd=""
fi

export GBS=$(expr $train_batch_size \* $num_gpu \* $num_accumulation_steps)
printf -v TAG "tf_bert_wikimed_1n_phase1_cased_%s_%s_gbs%d" "$cased" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"

echo "mpi command"
echo $mpi_command

$mpi python3 /workspace/bert/run_pretraining.py \
 --input_files_dir=$INPUT_FILES_DIR \
 --eval_files_dir=$EVAL_FILES_DIR \
 --output_dir=$CHECKPOINTS_DIR \
 --bert_config_file=$BERT_CONFIG \
 --do_train=True \
 --do_eval=True \
 --train_batch_size=$train_batch_size \
 --eval_batch_size=$eval_batch_size \
 --max_seq_length=128 \
 --max_predictions_per_seq=20 \
 --num_train_steps=$train_steps \
 --num_warmup_steps=$warmup_steps \
 --save_checkpoints_steps=$save_checkpoint_steps \
 --num_accumulation_steps=$num_accumulation_steps \
 --learning_rate=$learning_rate \
 --report_loss \
 $use_hvd $use_fp16 $use_xla_tag \
 --init_checkpoint=$INIT_CHECKPOINT |& tee $LOGFILE
