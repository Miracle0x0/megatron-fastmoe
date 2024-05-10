#!/usr/bin/env bash

echo "Launching GPT training job"

FMOE_FASTER_SCHEDULE_ENABLE=1 ./examples/pretrain_gpt_distributed_with_mp.sh
