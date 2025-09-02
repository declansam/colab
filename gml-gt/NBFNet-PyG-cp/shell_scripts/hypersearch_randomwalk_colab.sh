#!/bin/bash

port=$(python find_free_port.py)
# config="config/transductive/wn18rr-kgsampling-hypersearch.yaml"
config="config/transductive/fb15k237-kgsampling-randomwalk-hypersearch.yaml"

for i in {0..29}
do
    python script/hypersearch_randomwalk.py -c "$config" --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=3 --master_port="$port" script/instance_explanation.py -c "$config" --hyper_search
done