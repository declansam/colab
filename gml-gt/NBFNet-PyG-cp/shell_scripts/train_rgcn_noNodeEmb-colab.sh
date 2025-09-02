#!/bin/bash

config="config/hetero/synthetic-no-node-emb-colab.yaml"

port=$(python find_free_port.py)

# echo Port Found!
for i in {0..29}
do
    python script/hypersearch_rgcn_noNodeEmb.py -c "$config" --hyper_run_id="$i"
    python script/run_synthetic.py --hyper_search -c "$config"

done