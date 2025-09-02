import torch
import os.path as osp

if __name__ == "__main__":

    dirs = [
        "/scratch/dd115/Graph-Transformer/NBFNet-PyG/explanation/PowerLink/NBFNet/FB15k-237/p0",
        "/scratch/dd115/Graph-Transformer/NBFNet-PyG/explanation/PowerLink/NBFNet/FB15k-237/p1",
        "/scratch/dd115/Graph-Transformer/NBFNet-PyG/explanation/PowerLink/NBFNet/FB15k-237/p2",
        "/scratch/dd115/Graph-Transformer/NBFNet-PyG/explanation/PowerLink/NBFNet/FB15k-237/p3",
    ]

    base_path = "test_explanations_factual_eval_hard_edge_mask_top_k"
    save_dir = "/scratch/dd115/Graph-Transformer/NBFNet-PyG/explanation/PowerLink/NBFNet/FB15k-237/saved_explanations"
    for k in [25, 50, 75, 100, 300, 500, 1000]:
        path = base_path + f"_{k}.pt"
        expl = torch.load(osp.join(dirs[0], path), map_location=torch.device("cpu"))
        for dir in dirs[1:]:
            expl = torch.cat(
                (
                    expl,
                    torch.load(osp.join(dir, path), map_location=torch.device("cpu")),
                )
            )

        torch.save(expl, osp.join(save_dir, path))
