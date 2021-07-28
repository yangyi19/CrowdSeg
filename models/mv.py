import os
import numpy as np
import torch



class MvFusion(object):
    def __call__(self, ori_seg):
        '''
        :param ori_seg: tensor (batch, c, h, w)
        :return: ndarray (h, w, c)
        '''
        mask_sum = torch.sum(ori_seg, 1)
        # print('masks_sum shape:{}'.format(mask_sum.shape))
        result = (mask_sum > ori_seg.shape[1] / 2)
        return result.permute(1, 2, 0).numpy().astype(np.uint8)

    def __str__(self):
        return 'mv'



