#!/bin/bash

port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb_edgeraw.py -c config/edge_rw/fb15k237-rawexplainer-hyp-jub2.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/edge_rw/fb15k237-rawexplainer-hyp-jub2.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-jub1.yaml
done