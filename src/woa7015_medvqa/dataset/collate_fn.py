import torch


def collate_fn(batch):
    imgs, qs, as_, idxs = zip(*batch)
    imgs = torch.stack(imgs)
    m = max(x.size(0) for x in qs)
    q_pad = torch.zeros(len(qs), m, dtype=torch.long)
    for i, q in enumerate(qs):
        q_pad[i, : q.size(0)] = q
    return imgs, q_pad, torch.tensor(as_), torch.tensor(idxs)
