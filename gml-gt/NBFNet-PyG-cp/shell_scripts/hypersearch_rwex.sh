#!/bin/bash

port=$(python find_free_port.py)
# config="config/transductive/wn18rr-kgsampling-hypersearch.yaml"
config="config/transductive/wn18rr-kgsampling-rwexplainer-hypersearch.yaml"

for i in {0..9}
do
    # python script/hypersearch_set_config.py -c "$config" --hyper_run_id="$i"
    python script/hypersearch_rwexplainer.py -c "$config" --hyper_run_id="$i"
    # python -m torch.distributed.launch --nproc_per_node=3 --master_port="$port" script/kg_sampling.py -c "$config" --hyper_search
    python -m torch.distributed.launch --nproc_per_node=3 --master_port="$port" script/global_explanation.py -c "$config" --hyper_search
done