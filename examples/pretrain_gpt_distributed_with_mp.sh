#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=29504
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# DATA_PATH=<Specify path and file prefix>_text_document
# CHECKPOINT_PATH=<Specify path>

DATA_PREFIX=$DATA_PREFIX
DATA_PATH=$DATA_PREFIX/datasets/meg-gpt2_text_document
CHECKPOINT_PATH=$DATA_PREFIX/checkpoints/gpt2_fastmoe
LOG_PATH=$DATA_PREFIX/logs
VOCAB_FILE=$DATA_PREFIX/gpt2-vocab.json
MERGE_FILE=$DATA_PREFIX/gpt2-merges.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 8 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --fmoefy \
       --num-experts 2 \
       --fmoe-num-experts 2 \
       --balance-strategy gshard \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --no-contiguous-buffers-in-local-ddp \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       2>&1 | tee $CHECKPOINT_PATH/megatron_log.txt
