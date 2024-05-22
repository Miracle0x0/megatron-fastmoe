#!/usr/bin/env bash

echo "Launching BERT training job"

echo "EP_SIZE: 2"

FMOE_FASTER_SCHEDULE_ENABLE=1 ./examples/pretrain_bert_distributed_with_mp.sh 2


echo "EP_SIZE: 4"

FMOE_FASTER_SCHEDULE_ENABLE=1 ./examples/pretrain_bert_distributed_with_mp.sh 4


echo "EP_SIZE: 8"

FMOE_FASTER_SCHEDULE_ENABLE=1 ./examples/pretrain_bert_distributed_with_mp.sh 8

echo "Finished BERT training job"
