#!/bin/bash

# pip uninstall -y lxml

pip install -U \
    torch==2.1.2 \
    bitsandbytes \
    lm-eval[wandb,vllm]==0.4.2

pip install -U --no-build-isolation flash-attn