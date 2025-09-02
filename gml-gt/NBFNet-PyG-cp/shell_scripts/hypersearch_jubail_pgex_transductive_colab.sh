#!/bin/bash

port=$(python find_free_port.py)
config="config/transductive/fb15k237-kgsampling-pgexplainer-transductive-hypersearch-colab.yaml"

for i in {0..9}
do
    python script/hypersearch_pgexplainer_transductive.py -c "$config" --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation.py -c "$config" --hyper_search
done