"""
copy and paste form https://github.com/AiArt-Gao/HMEG/blob/a55ca678f6f72ec54b6364eb01d822f74cabd93e/data/crohme_box.py
"""

import numpy as np
import torch

from gryphgen.utils import MODELS


def to_triplet(batch):
    pair_batch = []
    for i, (node_types, edge_types, edges) in enumerate(batch):
        # gurd
        edges = np.asarray(edges, dtype=np.int64)  # np.object

        node_types = torch.as_tensor(node_types, dtype=torch.long)  # N
        edge_types = torch.as_tensor(edge_types, dtype=torch.long)  # E

        if len(edge_types) > 0:
            edges = torch.as_tensor(edges, dtype=torch.long)  # (2, E)

            E = edge_types.numel()
            triplets = torch.empty((E, 3), dtype=torch.long)
            triplets[:, 0] = edges[0]
            triplets[:, 1] = edge_types
            triplets[:, 2] = edges[1]
        else:
            triplets = torch.empty((0, 3), dtype=torch.long)

        pair_batch.append((node_types, triplets))

    return pair_batch


@MODELS.register_module()
def graph_preprocesser(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: REMOVED
    - triples: FloatTensor of shape (T, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    batch = to_triplet(batch)

    # batch is a list, and each element is (image, objs, boxes, triples)
    all_objs, all_triples = [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (objs, triples) in enumerate(batch):
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_objs = torch.cat(all_objs)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    return all_objs, all_triples, all_obj_to_img, all_triple_to_img
