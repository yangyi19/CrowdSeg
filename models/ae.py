import os

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from models.autoencoder import AutoEncoder
from common.helper import jaccard
from models.mv import MvFusion


class AeFusion(object):
    def __init__(self, model_num, max_epo=50):
        self.model_num = model_num
        self.ae = AutoEncoder(model_num)
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=0.01)
        self.loss_f = nn.MSELoss()
        self.mv = MvFusion()
        self.max_epo = max_epo
        self.loss = []

    def __call__(self, ori_seg):
        aeinput = ori_seg
        self.mv_mask = self.mv(ori_seg).squeeze()
        # train auto-encoder
        for ii in range(self.max_epo):
            self.ae.train()
            code, output = self.ae(aeinput)
            loss = self.loss_f(output, aeinput)
            self.optimizer.zero_grad()
            loss.backward()
            # print(loss)
            self.optimizer.step()
        self.loss.append(loss.item())
        code = code.detach().numpy().squeeze()
        # remove background pixels
        mask_1 = self.get_segmentation(code, code)
        y = code[mask_1 == 1]
        # get final result
        result = self.get_segmentation(y, code)
        return 1 - result

    def get_segmentation(self, mask, code):
        shape = code.shape
        kmeans = KMeans(n_clusters=2, random_state=0).fit(mask.reshape(-1, 1))
        z = kmeans.predict(code.reshape(-1, 1))
        mask = z.reshape(shape).astype(np.uint8)
        # result = (code > 0.2).astype(np.uint8)
        if jaccard(self.mv_mask, mask) < jaccard(self.mv_mask, 1 - mask):
            mask = 1 - mask
        return mask

    def __str__(self):
        return str(self.ae) + 'epo_{}'.format(self.max_epo)



