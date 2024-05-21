#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <method>"
    exit 1
fi

method=$1

echo "[BERT] Launching BERT training job"

sleep 1s

echo "[BERT] ===== EP_SIZE: 2 ====="

if [[ $method == "fastmoe" ]]; then
    FMOE_FASTER_SCHEDULE_ENABLE=1 ./examples/pretrain_bert_distributed_with_mp.sh 2 $method
else
    ./examples/pretrain_bert_distributed_with_mp.sh 2 $method
fi

sleep 5s

echo "[BERT] ===== EP_SIZE: 4 ====="

if [[ $method == "fastmoe" ]]; then
    FMOE_FASTER_SCHEDULE_ENABLE=1 ./examples/pretrain_bert_distributed_with_mp.sh 4 $method
else
    ./examples/pretrain_bert_distributed_with_mp.sh 4 $method
fi

sleep 5s

echo "[BERT] ===== EP_SIZE: 8 ====="

if [[ $method == "fastmoe" ]]; then
    FMOE_FASTER_SCHEDULE_ENABLE=1 ./examples/pretrain_bert_distributed_with_mp.sh 8 $method
else
    ./examples/pretrain_bert_distributed_with_mp.sh 8 $method
fi

echo "[BERT] Finished BERT training job"
