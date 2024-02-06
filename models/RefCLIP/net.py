# coding=utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.language_encoder import language_encoder
# from models.language_decoder import language_decoder
from models.visual_encoder import visual_encoder , process_yolov3_output
from models.RefCLIP.head import WeakREChead
from models.network_blocks import MultiScaleFusion
from models.tag_encoder import tag_encoder
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.color_indices = torch.tensor([6, 13, 231, 43, 191, 369, 194, 52, 110, 80, 125, 778, 673])
        self.select_num = __C.SELECT_NUM
        self.visual_encoder = visual_encoder(__C).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)
        self.tag_encoder = tag_encoder(__C, pretrained_emb, token_size) # 创建tag专用的encoder
        # self.lang_decoder = language_decoder(__C)
        self.linear_decoder = nn.Sequential(
            nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        )
        self.linear_vs = nn.Linear(1024, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_vs_pos = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE) # 加入visual_position_embedding 专用线性层
        self.linear_tag = nn.Linear(__C.HIDDEN_SIZE,__C.HIDDEN_SIZE) # 创建tag专用的Linear层
        self.soft_weights=nn.Linear(512,2) # 动态加权
        self.gumbel_softmax = F.gumbel_softmax
        self.head = WeakREChead(__C)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=1024, scaled=True)
        self.class_num = __C.CLASS_NUM
        if __C.VIS_FREEZE:
            self.frozen(self.visual_encoder)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        # Vision and Language Encoding
        with torch.no_grad():
            boxes_all, x_, boxes_sml = self.visual_encoder(x)
        y_ = self.lang_encoder(y)
        
        # Vision Multi Scale Fusion
        s, m, l = x_
        x_input = [l, m, s]
        l_new, m_new, s_new = self.multi_scale_manner(x_input)
        x_ = [s_new, m_new, l_new]

        # Anchor Selection
        boxes_sml_new = []
        mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True)
        mean_i = mean_i.squeeze(2)[:, :, 4]
        vals, indices = mean_i.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
        bs, gridnum, anncornum, ch = boxes_sml[0].shape
        bs_, selnum = indices.shape
        
        box_sml_new = boxes_sml[0].masked_select(
            torch.zeros(bs, gridnum).to(boxes_sml[0].device).scatter(1, indices, 1).bool().unsqueeze(2).unsqueeze(
                3).expand(bs, gridnum, anncornum, ch)).contiguous().view(bs, selnum, anncornum, ch)
        boxes_sml_new.append(box_sml_new) 
        ###选出筛选过后的锚点的类别，传出那个类别的词索引然后投入encoder。process_yolov3_output在visual encoder文件里面
        tag_feature, position_embedding = process_yolov3_output(boxes_sml_new[0],x) #[64,17,2], [64,17,512]
        bssize,num_anchor,ft = tag_feature.shape
        tag_feature = tag_feature.view(bssize*num_anchor,ft) #[64*17,1]
        tag_emb = self.tag_encoder(tag_feature) #用专门的tag encoder提取特征
        # with torch.no_grad():
        #     tag_emb = self.lang_encoder(tag_feature) #共用lang encoder提取特征,freeze encoder
        batchsize, dim, h, w = x_[0].size()
        i_new = x_[0].view(batchsize, dim, h * w).permute(0, 2, 1)
        bs, gridnum, ch = i_new.shape
        i_new = i_new.masked_select(
            torch.zeros(bs, gridnum).to(i_new.device).scatter(1, indices, 1).
                bool().unsqueeze(2).expand(bs, gridnum,ch)).contiguous().view(bs, selnum, ch)
        language_emb = y_['flat_lang_feat'].unsqueeze(1) #[64, 1, 512]
        tag_emb = tag_emb['flat_lang_feat'].view(bssize,num_anchor,512) #[64, 17, 512]
        # Anchor-based Contrastive Learning
        language_emb = self.linear_ts(language_emb) #[64,1,512]
        visual_emb = self.linear_vs(i_new) #[64,17,512]        
        # Dynamic Weighted Fusion
        weights = self.soft_weights(visual_emb + tag_emb + position_embedding) #[64,17,2] weights 用visual+tag一起算,
        logsoftmax = nn.LogSoftmax(dim=-1)
        weights= logsoftmax(weights/0.03) # softmax normalize       
        visual_emb = self.linear_vs_pos(visual_emb * weights[:,:,0,None]) + \
            self.linear_tag(tag_emb * weights[:,:,1,None] ) + position_embedding 
        # reconstruct the language embedding from the visual embedding and add reconstruction loss
        # loss_a = 0.8
        if self.training:
            contrastive_loss,max_sim_vis_emb = self.head(visual_emb, language_emb)
            # recon_lang_emb = self.linear_decoder(max_sim_vis_emb)
            # recon_loss = reconstruction_loss(recon_lang_emb, language_emb.detach())
            # total_loss = loss_a * contrastive_loss + (1-loss_a) * recon_loss
            return contrastive_loss
        else:
            predictions_s, similiarity_score = self.head(visual_emb, language_emb)
            predictions_list = [predictions_s]
            box_pred = get_boxes(boxes_sml_new, predictions_list,self.class_num)
            return box_pred,similiarity_score

def get_boxes(boxes_sml, predictionslist,class_num):
    batchsize = predictionslist[0].size()[0]
    pred = []
    for i in range(len(predictionslist)):
        mask = predictionslist[i].squeeze(1)
        masked_pred = boxes_sml[i][mask]
        refined_pred = masked_pred.view(batchsize, -1, class_num+5)
        refined_pred[:, :, 0] = refined_pred[:, :, 0] - refined_pred[:, :, 2] / 2
        refined_pred[:, :, 1] = refined_pred[:, :, 1] - refined_pred[:, :, 3] / 2
        refined_pred[:, :, 2] = refined_pred[:, :, 0] + refined_pred[:, :, 2]
        refined_pred[:, :, 3] = refined_pred[:, :, 1] + refined_pred[:, :, 3]
        pred.append(refined_pred.data)
    boxes = torch.cat(pred, 1)
    score = boxes[:, :, 4]
    max_score, ind = torch.max(score, -1)
    ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
    box_new = torch.gather(boxes, 1, ind_new)
    return box_new
    
def reconstruction_loss(generated_lang_emb, original_lang_emb):
    return F.mse_loss(generated_lang_emb, original_lang_emb)
