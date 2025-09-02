# pagelink
python script/instance_explanation_optim.py -c config/new_explain/wn18rr-pagelink-jubail.yaml --gpus [0] --use_wandb yes --train_config config/new_nbfnet/wn18rr-finetune-jub4.yaml


# pgexplainer
port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/wn18rr-pgexplainer-jubail.yaml --use_wandb yes --gpus [0,1] --train_config config/new_nbfnet/wn18rr-finetune-jub4.yaml


# gnnexplainer
port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/instance_explanation_optim.py -c config/new_explain/wn18rr-gnnexplainer-ola.yaml --gpus [0,1] --use_wandb yes --train_config config/new_nbfnet/wn18rr-finetune-ola5.yaml

# powerlink
python script/instance_explanation_optim.py -c config/new_explain/wn18rr-powerlink-dd.yaml --gpus [0] --use_wandb yes --train_config config/new_nbfnet/wn18rr-finetune-dd.yaml

# gnnexplainer fb
port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/instance_explanation_optim.py -c config/new_explain/fb15k237-gnnexplainer-ola.yaml --gpus [0,1] --use_wandb yes --train_config config/new_nbfnet/fb15k237-finetune-ola.yaml

# pgexplainer fb
port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-pgexplainer-ola.yaml --use_wandb yes --gpus [0,1,2,3] --train_config config/new_nbfnet/fb15k237-finetune-ola.yaml

# pgexplainer fb hyp search
port=$(python script/find_free_port.py)
config=config/new_explain/fb15k237-pgexplainer-hyp-ola.yaml
for i in {0..9}
do
    python script_hypersearch/hypersearch_pgexplainer_inductive.py -c "$config" --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c "$config" --hyper_search
# hyp search rawexp

port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb.py -c config/new_explain/fb15k237-rawexplainer-hyp-jub1.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-hyp-jub1.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-jub1.yaml
done

# hyp search rawexp

port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola2.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola2.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-ola.yaml
done


# pagelink
python script/instance_explanation_optim.py -c config/new_explain/fb15k237-pagelink-jub-p0.yaml --gpus [0] --use_wandb yes


port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola5.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola5.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-ola5.yaml
done

port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola3.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola3.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-ola.yaml
done


port=$(python script/find_free_port.py)

for th in 0.2 0.5 0.7; do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-inf-lam1a.yaml --gpus [0,1,2,3] --use_wandb no --train_config config/new_nbfnet/fb15k237-finetune-lam1.yaml
done

port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-inf-lam1a.yaml --gpus [0,1,2,3] --use_wandb no --train_config config/new_nbfnet/fb15k237-finetune-lam1.yaml


python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-inf-colabc.yaml --gpus [0,1,2,3] --use_wandb yes --train_config config/new_nbfnet/fb15k237-finetune-colab.yaml

port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-inf-olab2.yaml --gpus [0,1,2,3] --use_wandb yes --train_config config/new_nbfnet/fb15k237-finetune-inf-ola.yaml

# EdgeRAWExplainer Hyp Search

port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb_edgeraw.py -c config/edge_rw/fb15k237-rawexplainer-hyp-jub1.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/edge_rw/fb15k237-rawexplainer-hyp-jub1.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-jub1.yaml
done

port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb_edgeraw.py -c config/edge_rw/fb15k237-rawexplainer-hyp-colab.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/edge_rw/fb15k237-rawexplainer-hyp-colab.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-colab.yaml
done

# edge random walk fb
port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/instance_explanation_optim.py -c config/edge_rw/fb15k237-edgerandomwalk-colab.yaml --gpus [0,1,2,3] --use_wandb yes --train_config config/new_nbfnet/fb15k237-finetune-colab.yaml
