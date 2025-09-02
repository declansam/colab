#!/bin/bash

port=$(python script/find_free_port.py)
config=config/new_explain/fb15k237-rawexplainer-hyp-lam1.yaml

for i in {0..19}
do
    python script_hypersearch/hypersearch_rawexplainer_fb15k237.py -c "$config" --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c "$config" --hyper_search --train_config config/new_explain/fb15k237-rawexplainer-hyp-lam1.yaml
done