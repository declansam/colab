# Knowledge Graph Sampling
All of the scripts can also be launched with DDP, unless otherwise specified.


## Experiments on Synthetic datasets

To train the GNN model with 1 GPU and log the training on wandb
```bash
python script/run_synthetic_auc.py -c config/hetero/{config_file} --gpus [0] --use_wandb yes
```
Refer to `config/hetero/synthetic-nbfnet.yaml` as an example.
Refer to `config/hetero/synthetic-nbfnet-noGT.yaml` as an example for training a GNN on a dataset with ground truth removed.

**Note: DDP is slow in this setup, but dataset is small enough that 1 GPU is enough.**

To evaluate the trained GNN on removing the ground truth / keeping the ground truth but varying noise / randomly dropped edges, edit the `eval` section of the config file. See `config/hetero/synthetic-inf.yaml` as an example.
```bash
python script/run_synthetic_auc.py -c config/hetero/{config_file} --gpus [0] --use_wandb no
```

**To Do: Explanation experiments**

## Experiments on WN18RR, FB15k-237

### GNN Training

**Default Training Setup**

To train a GNN in the default setup, run the following command.
```bash
python script/train_nbfnet.py -c config/transductive/{config_file} --gpus [0] --use_wandb yes
```
This script supports DDP.

For training, make sure to specify `cfg.model.use_pyg_propagation` as `yes`. Without it, the default pipeline will use RSPMM kernel, but this leads to inconsistencies at the explanation stage. 

**Training Robust GNNs**

Refer to `config/transductive/wn18rr-0.2b-rwdropout.yaml` for advanced randomized edge drop with random walk.

To train a robust GNN (GNN trained with randomized edge dropping), use either the default training setup with specification for `cfg.model.randomized_edge_drop` (refer to `config/transductive/wn18rr-0.2b.yaml`), or the following command for advanced randomized edge drops (random walk, distance-based) 


**Note: This setup is still in construction.**


**Deprecated: Training GNNs only on the ego-network**

To train a GNN with only the ego-network, run the following:

```bash
python script/run_ego.py -c config/transductive/{config_file} --gpus [0] --use_wandb yes
```

Refer to `wn18rr-ego-0.2b.yaml`.

### Explanations

The explainers are split into two categories: 
1. **Global Explainers** which include PGExplainer, RelPGExplainer, and RWExplainer
2. **Instance Explainers** which include GNNExplainer, PaGE-Link, and RandomWalk

**Running Global Explainers**

To run any of the global explainers, use the following command:

```bash
python script/global_explanation.py -c config/transductive/{config_file} --gpus [0] --use_wandb yes
```

Furthermore, to perform hyperparameter tuning, refer to `hypersearch_rwex.sh` as an example.


**Running Instance Explainers**

To run any of the instance explainers, use the following command:

```bash
python script/instance_explanation.py -c config/transductive/{config_file} --gpus [0] --use_wandb yes
```

Furthermore, to perform hyperparameter tuning, refer to `hypersearch_jubail_gnnex.sh`.

**Saving and Vizualizing the Explanation**

To save the explanation, run the explainers but specify `cfg.save_explanation` as `yes`. Saving the explanation cannot run with DDP.

To vizualize the explanation, refer to `vizualize_explanation.ipynb`.