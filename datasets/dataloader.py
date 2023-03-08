# coding=utf-8

import os
import cv2
import json, re, en_vectors_web_lg, random

import numpy as np

import torch
import torch.utils.data as Data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils.utils import label2yolobox


class RefCOCODataSet(Data.Dataset):
    def __init__(self, __C,split):
        super(RefCOCODataSet, self).__init__()
        self.__C = __C
        self.split=split
        assert  __C.DATASET in ['refcoco', 'refcoco+', 'refcocog','referit']
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
        total_refs_list=[]
        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]
        refs=[]
        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)
        for split in total_refs_list:
            for ann in total_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)

        self.image_path=__C.IMAGE_PATH[__C.DATASET]

        self.input_shape=__C.INPUT_SHAPE
        # Define run data size
        self.data_size = len(self.refs_anno)

        print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------
        # Tokenize
        self.token_to_ix,self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        self.max_token = __C.MAX_TOKEN
        if self.max_token == -1:
            self.max_token = max_token
        print('Max token length:', max_token, 'Trimmed to:', self.max_token)
        print('Finished!')
        print('')

        self.candidate_transforms ={}
        self.transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(__C.MEAN, __C.STD)])



    def tokenize(self, stat_refs_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    words = re.sub(
                        r"([.,'!?\"()*#:;])",
                        '',
                        ref.lower()
                    ).replace('-', ' ').replace('/', ' ').split()

                    if len(words) > max_token:
                        max_token = len(words)

                    for word in words:
                        if word not in token_to_ix:
                            token_to_ix[word] = len(token_to_ix)
                            if use_glove:
                                pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        ix_to_token={}
        for item in token_to_ix:
            ix_to_token[token_to_ix[item]]=item

        return token_to_ix, ix_to_token,pretrained_emb, max_token


    def proc_ref(self, ref, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ref.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_refs(self, idx):
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        ref=self.proc_ref(ref,self.token_to_ix,self.max_token)
        return ref

    def preprocess_info(self,img,box,iid,lr_flip=False):
        h, w, _ = img.shape
        imgsize=self.input_shape[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2
        img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        info_img = (h, w, nh, nw, dx, dy,iid)
        sized_box=label2yolobox(box,info_img,self.input_shape[0],lrflip=lr_flip)
        return sized,sized_box, info_img

    def load_img_feats(self, idx):
        img_path=None
        if self.__C.DATASET in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_path,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError
        image= cv2.imread(img_path)
        box=np.array([self.refs_anno[idx]['bbox']])
        return image,box,self.refs_anno[idx]['iid']

    def __getitem__(self, idx):

        ref_iter = self.load_refs(idx)
        image_iter,gt_box_iter,iid= self.load_img_feats(idx)
        image_iter = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
        ops=None
        if len(list(self.candidate_transforms.keys()))>0:
            ops = random.choices(list(self.candidate_transforms.keys()), k=1)[0]
        if ops is not None and ops!='RandomErasing':
            image_iter = self.candidate_transforms[ops](image=image_iter)['image']
        flip_box=False
        image_iter,box_iter,info_iter=self.preprocess_info(image_iter,gt_box_iter.copy(),iid,flip_box)
        return \
            torch.from_numpy(ref_iter).long(), \
            self.transforms(image_iter),\
            torch.from_numpy(box_iter).float(), \
            torch.from_numpy(gt_box_iter).float(), \
            np.array(info_iter)

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)

def loader(__C,dataset: torch.utils.data.Dataset, rank: int, shuffle,drop_last=False):
    if __C.MULTIPROCESSING_DISTRIBUTED:
        assert __C.BATCH_SIZE % len(__C.GPU) == 0
        assert __C.NUM_WORKER % len(__C.GPU) == 0
        assert dist.is_initialized()

        dist_sampler = DistributedSampler(dataset,
                                          num_replicas=__C.WORLD_SIZE,
                                          rank=rank)
        data_loader = DataLoader(dataset,
                                 batch_size=__C.BATCH_SIZE // len(__C.GPU),
                                 shuffle=shuffle,
                                 sampler=dist_sampler,
                                 num_workers=__C.NUM_WORKER //len(__C.GPU),
                                 pin_memory=False,
                                 drop_last=drop_last)
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=__C.BATCH_SIZE,
                                 shuffle=shuffle,
                                 num_workers=__C.NUM_WORKER,
                                 pin_memory=False,
                                 drop_last=drop_last)
    return data_loader






