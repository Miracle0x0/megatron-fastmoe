#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=2

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=29505
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")

# EP_SIZE=2
EP_SIZE=$1

METHOD=$2

# DATA_PATH=<Specify path and file prefix>_text_sentence
# VOCAB_FILE=<Specify path to vocab.txt>
# CHECKPOINT_PATH=<Specify path>

BERT_PREFIX=$BERT_PREFIX
DATA_PATH=$BERT_PREFIX/datasets/meg-bert_text_sentence
CHECKPOINT_PATH=$BERT_PREFIX/checkpoints/bert_${METHOD}_ep_${EP_SIZE}_${CURRENT_TIME}
LOG_PATH=$BERT_PREFIX/logs
VOCAB_FILE=$BERT_PREFIX/bert-large-uncased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 12 \
       --hidden-size 1024 \
       --num-attention-heads 8 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --fmoefy \
       --num-experts $EP_SIZE \
       --fmoe-num-experts $EP_SIZE \
       --balance-strategy gshard \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --no-contiguous-buffers-in-local-ddp \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       2>&1 | tee $LOG_PATH/bert_${METHOD}_ep_${EP_SIZE}.log
