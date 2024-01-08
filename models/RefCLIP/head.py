# coding=utf-8
import torch
import torch.nn as nn


def getContrast(vis_emb, lan_emb, tag_emb, a):
    # vis_emb.shape = [64,17,512], lan_emb.shape = [64,1,512], tag_emb = [64,17,512]
    #text-visual
    sim_map_vis = torch.einsum('avd, bqd -> baqv',vis_emb,lan_emb) #[64,64,1,17]
    #tag-text
    sim_map_tag = torch.einsum('avd, bqd -> baqv',tag_emb,lan_emb) #[64,64,1,17]
    sim_map= a*sim_map_vis+ (1-a)*sim_map_tag
    batchsize = sim_map.shape[0]
    max_sims,_ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True)
    max_sims = max_sims.squeeze(2)

    # Negative Anchor Augmentation
    max_sim_0,max_sim_1 = max_sims[...,0],max_sims[...,1]
    max_sim_1 = max_sim_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_1.device)).contiguous().view(batchsize,batchsize-1)
    new_logits = torch.cat([max_sim_0,max_sim_1],dim=1)

    target = torch.eye(batchsize).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)
    loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    return loss

def getPrediction(vis_emb, lan_emb, tag_emb, a):
    # 计算视觉特征和语言特征之间的相似度
    sim_map_vis = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)  
    # 计算标签特征和语言特征之间的相似度
    sim_map_tag = torch.einsum('bkd, byd -> byk', tag_emb, lan_emb) 
    
    sim_map= a*sim_map_vis+ (1-a)*sim_map_tag

    # 根据总的相似度分数进行预测
    maxval, v = sim_map.max(dim=2, keepdim=True)
    predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2, v.expand(sim_map.shape), 1).bool()

    return predictions


class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.8))

    def forward(self, fusion_fs,lan_fs, tag_fs):
        a = torch.sigmoid(self.a)
        if self.training:
            loss = getContrast(fusion_fs, lan_fs, tag_fs, a)
            return loss
        else:
            predictions = getPrediction(fusion_fs, lan_fs, tag_fs, a)
            return predictions










