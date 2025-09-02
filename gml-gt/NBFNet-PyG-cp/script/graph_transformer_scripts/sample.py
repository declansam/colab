import os.path as osp
import torch
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    dir_name = '/storage/ryoji/Graph-Transformer/NBFNet-PyG/topk'
    dataset = 'wn18rr'
    dir_path = osp.join(dir_name, dataset)

    train_top10_paths = torch.load(osp.join(dir_path, 'train_topk_paths.pt'))

    important_paths = {} # key: rel, value: dict

    count = 0
    for k, v in tqdm(train_top10_paths.items(), desc='Getting the Important Paths'):
        rel = k[1] # get the relation
        if len(v) == 0: # no paths
            # print('No paths')
            continue
        if v[0][-1] > 100: # if the predicted rank for this triple is more than 100, ignore.
            # print(f'rank is {v[0][-1]}, so ignoring')
            continue
        if rel not in important_paths.keys():
            important_paths[rel] = {} # add the rel to the dict
            # this second dict will be key: path pattern and value: list with 1st item the sum of weights and second the frequency
        
        for path_info in v:
            path = tuple(path_info[0])
            weight = path_info[1]
            if path not in important_paths[rel].keys():
                important_paths[rel][path] = [weight, 1]
            else:
                important_paths[rel][path][0] += weight
                important_paths[rel][path][1] += 1     
        
        count += 1

    print(f"In total: {count}/{len(train_top10_paths)} triples processed")
    torch.save(important_paths, osp.join(dir_path, 'imp_paths.pt'))

    sorted_paths = {}
    for k in tqdm(important_paths.keys(), desc='Sorting the Important Paths'):
        paths = important_paths[k]
        # s_paths = sorted(paths.items(), key=lambda item: item[1][0] / item[1][1], reverse=True) # by the average
        s_paths = sorted(paths.items(), key=lambda item: item[1][0], reverse=True) # by the summed weight (naturally penalizes the less frequent ones)
        sorted_paths[k] = s_paths
    
    torch.save(sorted_paths, osp.join(dir_path, 'sorted_imp_paths.pt'))





