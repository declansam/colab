#!/bin/bash

# #activate any environments if required
# source ~/.bashrc
# conda activate Ultra

port=$(python script/find_free_port.py)
config="config/explain/fb15k237-rawexplainer-hypersearch-lambda2.yaml"

for i in {0..29}
do
    python script_hypersearch/hypersearch_rawexplainer.py -c "$config" --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=8 --master_port="$port" script/global_explanation.py -c "$config" --hyper_search
done