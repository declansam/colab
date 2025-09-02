#!/bin/bash

port=$(python find_free_port.py)
# config="config/transductive/wn18rr-kgsampling-hypersearch.yaml"
config="config/hetero/synthetic-kgsampling-relpgexplainer-0.2b-dual-hypersearch.yaml"

for i in {0..29}
do
    python script_hypersearch/hypersearch_relpgexplainer_synthetic_dual.py -c "$config" --hyper_run_id="$i"
    # python script/hypersearch_rwexplainer.py -c "$config" --hyper_run_id="$i"
    # python -m torch.distributed.launch --nproc_per_node=3 --master_port="$port" script/kg_sampling.py -c "$config" --hyper_search
    python script/global_explanation.py -c "$config" --hyper_search
done