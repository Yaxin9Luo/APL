# coding=utf-8
import torch
import torch.nn as nn


def getContrast(vis_emb, lan_emb, tag_emb,a=1.0,b=1.0):
    # 计算视觉特征和语言特征之间的相似度 vis_lan_sim是[64,64,1,17]
    vis_lan_sim = torch.einsum('avd, bqd -> baqv', vis_emb, lan_emb)
    batchsize = vis_lan_sim.shape[0]
    vis_lan_max_sims, _ = vis_lan_sim.topk(k=2,dim=-1,largest=True, sorted=True) # torch.Size([64, 64, 1, 2])
    vis_lan_max_sims = vis_lan_max_sims.squeeze(2) #torch.Size([64, 64, 2])
    # 计算语言特征和标签特征之间的相似度 lan_tag_sim是[64,10816,1,1]
    lan_tag_sim = torch.einsum('avd, bqd -> baqv', tag_emb, lan_emb)
    lan_tag_max_sims, _ = lan_tag_sim.topk(k=64,dim=1,largest=True, sorted=True) # torch.Size([64, 64, 1, 1])
    # 扩展 lan_tag_max_sims 到 [64, 64, 1, 2]
    lan_tag_max_sims = torch.cat((lan_tag_max_sims, lan_tag_max_sims), dim=-1)  # 形状为 [64, 64, 1, 2]
    lan_tag_max_sims = lan_tag_max_sims.squeeze(2)#torch.Size([64, 64, 2])
    # visual-text Negative Anchor Augmentation
    vis_lan_max_sims_0, vis_lan_max_sims_1 = vis_lan_max_sims[..., 0], vis_lan_max_sims[..., 1]
    vis_lan_max_sims_1 = vis_lan_max_sims_1.masked_select(~torch.eye(batchsize).bool().to(vis_lan_max_sims_1.device)).contiguous().view(batchsize, batchsize - 1)
    new_logits_visual = torch.cat([vis_lan_max_sims_0, vis_lan_max_sims_1], dim=1) # shape of logit visual: torch.Size([64, 127])
    # tag-text Negative Anchor Augmentation
    lan_tag_max_sims_0, lan_tag_max_sims_1 = lan_tag_max_sims[..., 0], lan_tag_max_sims[..., 1]
    lan_tag_max_sims_1 = lan_tag_max_sims_1.masked_select(~torch.eye(batchsize).bool().to(lan_tag_max_sims_1.device)).contiguous().view(batchsize, batchsize - 1)
    new_logits_tag = torch.cat([lan_tag_max_sims_0, lan_tag_max_sims_1], dim=1)

    #加入text-tag的loss
    target = torch.eye(batchsize).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)
    loss_visual = nn.CrossEntropyLoss(reduction="mean")(new_logits_visual, target_pred)
    loss_tag = nn.CrossEntropyLoss(reduction="mean")(new_logits_tag, target_pred)
    loss = a*loss_visual + b*loss_tag
    return loss

def getPrediction(vis_emb, lan_emb, tag_emb):
    # 计算视觉特征和语言特征之间的相似度
    vis_lan_sim = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)  # [64, 1, 17]

    # 计算标签特征和语言特征之间的相似度
    lan_tag_sim = torch.einsum('bkd, ckd -> bkc', lan_emb, tag_emb)  # [64, 1, 10816]

    # 从 lan_tag_sim 中选出每个批次中最大的17个值
    top_values, top_indices = lan_tag_sim.topk(17, dim=-1)  # top_values: [64, 1, 17], top_indices: [64, 1, 17]

    # 结合视觉和标签的相似度
    total_sim = vis_lan_sim + top_values

    # 根据总的相似度分数进行预测
    maxval, v = total_sim.max(dim=2, keepdim=True)
    predictions = torch.zeros_like(total_sim).to(total_sim.device).scatter(2, v.expand(total_sim.shape), 1).bool()

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










