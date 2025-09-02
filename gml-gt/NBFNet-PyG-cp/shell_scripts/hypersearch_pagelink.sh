#!/bin/bash

port=$(python find_free_port.py)
# config="config/transductive/wn18rr-kgsampling-hypersearch.yaml"
config="config/transductive/wn18rr-kgsampling-pagelink-hypersearch.yaml"

for i in {0..9}
do
    # python script/hypersearch_set_config.py -c "$config" --hyper_run_id="$i"
    python script/hypersearch_pagelink.py -c "$config" --hyper_run_id="$i"
    python script/instance_explanation.py -c "$config" --hyper_search
done