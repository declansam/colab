#!/bin/bash

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

port=$(python script/find_free_port.py)
# config="config/transductive/wn18rr-kgsampling-hypersearch.yaml"
config="config/explain/wn18rr-rawexplainer-hypersearch-jubail.yaml"

for i in {0..29}
do
    # python script/hypersearch_set_config.py -c "$config" --hyper_run_id="$i"
    python script_hypersearch/hypersearch_rawexplainer.py -c "$config" --hyper_run_id="$i"
    # python -m torch.distributed.launch --nproc_per_node=3 --master_port="$port" script/kg_sampling.py -c "$config" --hyper_search
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/global_explanation.py -c "$config" --hyper_search
done