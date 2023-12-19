# coding=utf-8
import torch
import torch.nn as nn


def getContrast(vis_emb, lan_emb):
    sim_map = torch.einsum('avd, bqd -> baqv',vis_emb,lan_emb)
    # print(max_sims.shape)
    batchsize = sim_map.shape[0]
    # Get the top 2 similarity scores for each anchor across all words
    max_sims, _ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True)
    max_sims = max_sims.squeeze(-1)  # Remove the last dimension
    
    # max_sims,_ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True) sentence origin version
    # max_sims = max_sims.squeeze(2)

    # Negative Anchor Augmentation
    max_sim_0,max_sim_1 = max_sims[...,0],max_sims[...,1]
    max_sim_0 = max_sim_0.max(dim=-1).values # Compute the maximum similarity score for each anchor across all words
    max_sim_1 = max_sim_1.max(dim=-1).values# Compute the second largest similarity score for each anchor across all words
    # print(max_sim_0.shape)
    # print(max_sim_1.shape)
    # print("batchsize:", batchsize)
    # print("max_sim_1 shape before masking:", max_sim_1.shape)
    # print("Mask shape:", (~torch.eye(batchsize).bool().to(max_sim_1.device)).shape)

    max_sim_1 = max_sim_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_1.device)).contiguous().view(batchsize,batchsize-1)
    new_logits = torch.cat([max_sim_0,max_sim_1],dim=1)

    target = torch.eye(batchsize).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)
    loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    return loss

def getPrediction(vis_emb, lan_emb):
    sim_map = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    max_sim_per_word, _ = sim_map.max(dim=-1)
    # print(max_sim_per_word.shape)
    maxval, v = max_sim_per_word.max(dim=1, keepdim=True)
    # Generate predictions based on the maximum similarity scores
    predictions = torch.zeros_like(max_sim_per_word).to(max_sim_per_word.device).scatter(1, v, 1).bool()
    # predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2,v.expand(sim_map.shape), 1).bool() sentence level
    return predictions

class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()

    def forward(self, vis_fs,lan_fs):
        if self.training:
            loss = getContrast(vis_fs, lan_fs)
            return loss
        else:
            predictions = getPrediction(vis_fs, lan_fs)
            return predictions










