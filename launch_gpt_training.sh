#!/usr/bin/env bash

echo "Launching GPT training job"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <EP_SIZE> $1 <method>"
    exit
fi

EP_SIZE=$1
method=$2

echo "EP_SIZE: ${EP_SIZE}"

FMOE_FASTER_SCHEDULE_ENABLE=1 ./examples/pretrain_gpt_distributed_with_mp.sh $EP_SIZE $method

echo "Finish GPT training job"
