# coding=utf-8
import torch
import torch.nn as nn


def getContrast(vis_emb, lan_emb,tag_emb):
    sim_map = torch.einsum('avd, bqd -> baqv',vis_emb,lan_emb)
    # Compute similarities between text and tag embeddings
    print("lan_emb", lan_emb.shape)
    print("tag_emb", tag_emb.shape)
    sim_map_lt = torch.einsum('avd, bqd -> baqv', lan_emb, tag_emb)
    batchsize = sim_map.shape[0]
    #select the max score from word level
    sim_map,_=sim_map.topk(k=1, dim=2, largest=True, sorted=True)
    # print("sim_map",sim_map.shape)
    max_sims,_ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True) 
    # print("max_sims",max_sims.shape)
    max_sims = max_sims.squeeze(2)
    # print("squeezed max_sims",max_sims.shape)
    
    # Process Language-Tag similarity
    max_sims_lt, _ = sim_map_lt.topk(k=2, dim=-1, largest=True, sorted=True)

    # Negative Anchor Augmentation
    max_sim_0,max_sim_1 = max_sims[...,0],max_sims[...,1]
    max_sim_lt_0, max_sim_lt_1 = max_sims_lt[..., 0], max_sims_lt[..., 1]

    # print(max_sim_0.shape)
    # print(max_sim_1.shape)
    # print("batchsize:", batchsize)
    # print("max_sim_1 shape before masking:", max_sim_1.shape)
    # print("Mask shape:", (~torch.eye(batchsize).bool().to(max_sim_1.device)).shape)

    max_sim_1 = max_sim_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_1.device)).contiguous().view(batchsize,batchsize-1)
    max_sim_lt_1 = max_sim_lt_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_lt_1.device)).contiguous().view(batchsize, batchsize - 1)

    new_logits = torch.cat([max_sim_0,max_sim_1],dim=1)
    new_logits_lt = torch.cat([max_sim_lt_0.unsqueeze(1), max_sim_lt_1], dim=1)
    new_logits = (new_logits + new_logits_lt) / 2

    target = torch.eye(batchsize).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)
    loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    return loss

def getPrediction(vis_emb, lan_emb,tag_emb):
    sim_map = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    # Compute similarities between visual and tag embeddings
    sim_map_lt = torch.einsum('bkd, byd -> byk', lan_emb, tag_emb)
    #select the max score from word level
    sim_map,_=sim_map.topk(k=1, dim=1, largest=True, sorted=True)
    #select max score of text-tag
    sim_map_lt, _ = sim_map_lt.topk(k=1, dim=1, largest=True, sorted=True)
    # Combine the similarities (e.g., averaging)
    sim_map = (sim_map + sim_map_lt) / 2

    maxval, v = sim_map.max(dim=2, keepdim=True)
    predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2,v.expand(sim_map.shape), 1).bool()
    return predictions

class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()

    def forward(self, vis_fs,lan_fs,tag_fs):
        if self.training:
            loss = getContrast(vis_fs, lan_fs,tag_fs)
            return loss
        else:
            predictions = getPrediction(vis_fs, lan_fs, tag_fs)
            return predictions










