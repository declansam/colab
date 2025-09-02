#!/bin/bash

# python script/instance_explanation_optim.py -c config/new_explain/wn18rr-pagelink-jubail.yaml --gpus [0] --use_wandb yes --train_config config/new_nbfnet/wn18rr-finetune-jub4.yaml
python script/instance_explanation_optim.py -c config/new_explain/fb15k237-pagelink-jub-p0.yaml --gpus [0] --use_wandb yes