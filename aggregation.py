import os
import cv2
from torch.utils.data import DataLoader

from dataset.crowd_dataset import CrowdDataset
from models.ae import AeFusion


def get_str(para_name, value_list_):
    l = [str(x) for x in value_list_]
    value_str = '_'.join(l)
    return '_'.join([para_name, value_str])


def init_store_path(params):
    inter_num = get_str('in', params['inter_num'])
    thres = get_str('th', params['thres'])
    ss = get_str('ss', params['super_size'])
    model = get_str('m', params['model'])
    tem = '_'.join([inter_num, thres, ss])
    store_path = os.path.join(params['store_root'], params['dataset'], params['fusion_model'], model, tem)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    print('store path:{}'.format(store_path))
    return store_path


class Fuse(object):
    def __init__(self, dataset, model, store_root):
        self.loader = DataLoader(dataset=dataset)
        self.model = model
        self.store_root = store_root
        self.store_path = os.path.join(self.store_root, str(dataset))
        print('store path:{}'.format(self.store_path))
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

    def __call__(self, *args, **kwargs):
        for ii, data in enumerate(self.loader):
            label = data['label']
            file_name = data['name'][0]
            fusion_label = self.model(label)
            self.store_result(fusion_label, file_name)
            # if ii % 10 == 0:
            # print('第{}个完成'.format(ii))
        print('done')
        return self.model

    def store_result(self, fusion_label, file_name):
        path = os.path.join(self.store_path, file_name)
        cv2.imwrite(path, (1 - fusion_label) * 255)


if __name__ == '__main__':
    # dataset
    dataset_set = [3, 4]
    # worker id
    worker_1 = [1, 2, 3, 4, 5, 6, 7]
    worker_2 = [1, 2, 3, 4, 5]
    worker_3 = [1, 2, 3]
    worker_id_set_set = [worker_1, worker_2, worker_3]
    # train epoch
    max_epo = 10
    for worker_id_set in worker_id_set_set:
        for dataset in dataset_set:
            # aggregation method
            fusion_m = AeFusion(model_num=len(worker_id_set), max_epo=max_epo)
            # store path of aggregation result
            store_root = os.path.join('.', 'aggregation_result', 'BU-BIL-resize', str(fusion_m))
            # segmentations of crowd workers
            dataset = CrowdDataset(dataset, worker_id_set)
            fuse = Fuse(dataset, fusion_m, store_root)
            # pretrain
            print('pretain')
            fuse()
            # aggregation
            print('aggregation')
            fuse()
