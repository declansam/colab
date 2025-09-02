import pandas as pd
import torch
import os
from collections import defaultdict


if __name__ == "__main__":
    working_dir = "/scratch/dd115/Graph-Transformer/NBFNet-PyG/explanation/PowerLink/NBFNet/FB15k-237/p2"
    eval_type = "factual_eval"
    save_explanation = True
    for r in [25, 50, 75, 100, 300, 500, 1000]:
        # make sure that all the result has been written to disk already.
        # * Aggr statistics across devices *

        # load the results from disk and combine.
        all_stats = defaultdict(list)
        all_explanations = []
        dir_path = os.path.join(working_dir, "saved_results")

        for i in range(11):
            save_path = os.path.join(dir_path, f"result_{eval_type}_{r}_{i}.pt")
            result = torch.load(save_path, map_location=torch.device("cpu"))
            for key, var in result.items():
                all_stats[key].append(var)

            if save_explanation:
                save_path = os.path.join(
                    dir_path, f"explanation_{eval_type}_{r}_{i}.pt"
                )
                explanation = torch.load(save_path, map_location=torch.device("cpu"))
                all_explanations.append(explanation)

        for key, var in all_stats.items():
            all_stats[key] = torch.cat(var)

        if save_explanation:
            all_explanations = torch.cat(all_explanations)
            additional_info = f"_{eval_type}_hard_edge_mask_top_k_{r}"
            torch.save(
                all_explanations,
                os.path.join(working_dir, f"test_explanations{additional_info}.pt"),
            )
