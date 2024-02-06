# coding=utf-8
import torch
import torch.nn as nn


def getContrast(vis_emb, lan_emb):
    # vis_emb.shape = [64,17,512], lan_emb.shape = [64,1,512], tag_emb = [64,17,512]
    #text-visual
    sim_map_vis = torch.einsum('avd, bqd -> baqv',vis_emb,lan_emb) #[64,64,1,17]
    select_anchor = torch.einsum('avd, bqd -> bqv',vis_emb,lan_emb)#[64,1,17]
    sim_map= sim_map_vis
    batchsize = sim_map.shape[0]
    max_sims,index = sim_map.topk(k=2, dim=-1, largest=True, sorted=True) #index = [64,64,1,2]
    _,top_index = select_anchor.topk(k=1, dim=-1, largest=True, sorted=True) # index = [64,1,1]
    max_sim_vis_emb = torch.gather(vis_emb, 1, top_index.expand(-1, -1, vis_emb.shape[-1]))
    max_sims = max_sims.squeeze(2)
    top_vs_emb = 1
    # Negative Anchor Augmentation
    max_sim_0,max_sim_1 = max_sims[...,0],max_sims[...,1]
    max_sim_1 = max_sim_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_1.device)).contiguous().view(batchsize,batchsize-1)
    new_logits = torch.cat([max_sim_0,max_sim_1],dim=1)
    target = torch.eye(batchsize).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)
    loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    return loss,max_sim_vis_emb

def getPrediction(vis_emb, lan_emb):
    # 计算视觉特征和语言特征之间的相似度
    sim_map_vis = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)  
    sim_map= sim_map_vis
    # 根据总的相似度分数进行预测
    maxval, v = sim_map.max(dim=2, keepdim=True)
    predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2, v.expand(sim_map.shape), 1).bool()

    return predictions, maxval.squeeze()


class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()
    def forward(self, fusion_fs,lan_fs):
        if self.training:
            loss = getContrast(fusion_fs, lan_fs)
            return loss
        else:
            predictions = getPrediction(fusion_fs, lan_fs)
            return predictions










