import os
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms



class CrowdDataset(Dataset):
    root = '.\BU-BIL-resize'

    def __init__(self, dataset_id, worker_id_set, Transform=transforms.ToTensor()):
        self.dataset_id = dataset_id
        self.worker_id_set = worker_id_set
        # root path of dataset, containing \RawImages directory
        self.root_path = os.path.join(self.root, 'BU-BIL_Dataset{}'.format(dataset_id))
        # path of worker's segmentation result, containing .png images
        self.worker_path_set = self.get_worker_path(dataset_id, worker_id_set)
        self.worker_num = len(worker_id_set)
        # 所有图片和标签的路径列表
        self.gts_path_list = self.get_file_path_list('GoldStandard')
        # self.labels_path_list = self.get_file_path_list('labels')
        self.transform = Transform

    def __getitem__(self, index):
        # 2：根据index取得相应的一幅图像，一幅标签的路径
        gt_path = self.gts_path_list[index]

        # 3：将图片和label读出。“L”表示灰度图，也可以填“RGB”
        gt = cv2.imread(gt_path, flags=cv2.IMREAD_GRAYSCALE)

        label = self.get_label(gt_path)

        # 4：tansform 参数一般为 transforms.ToTensor()，意思是上步image,label 转换为 tensor 类型
        if self.transform is not None:
            gt = self.transform(gt)
            label = self.transform(label)
        return dict(gt=1 - gt, label=1 - label, name=self.name)

    def get_file_path_list(self, images):
        '''
        得到指定目录 images 下的所有文件的路径的列表
        '''
        src_dir = os.path.join(self.root_path, images)
        image_path_list = glob(src_dir + '\*')
        return image_path_list

    def get_label(self, image_path):
        '''
        return the label path of the image stored in image_path
        '''
        (filepath, tempfilename) = os.path.split(image_path)
        label_name = tempfilename
        self.name = tempfilename
        label_set = []
        for worker_id in range(self.worker_num):
            path = os.path.join(self.worker_path_set[worker_id], label_name)
            label = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
            label_set.append(label)
        crowd_label = np.stack(label_set, axis=-1)
        return crowd_label

    def get_worker_path(self, dataset_id, worker_id_set):
        worker_path_set = []
        worker_id_str = []
        for ii in worker_id_set:
            worker = 'CrowdsourcedAnnotations\mturk_annotation{}-{}'.format(dataset_id, ii)
            path = os.path.join(self.root_path, worker)
            worker_path_set.append(path)
            worker_id_str.append(str(ii))
        self.worker_id = '_'.join(worker_id_str)
        return worker_path_set

    def __len__(self):
        return len(self.gts_path_list)

    def __str__(self):
        return os.path.join('BU-BIL_Dataset{}'.format(self.dataset_id), 'crowd', self.worker_id)


if __name__ == '__main__':
    root_path = '..\BU-BIL_Dataset1'
    worker_path = 'CrowdsourcedAnnotations\mturk_annotation1-1'
    dataset_id = 1
    worker_id_set = [1, 2, 3, 4]
    dataset = CrowdDataset(dataset_id, worker_id_set)
    dataloader = DataLoader(dataset=dataset)
    for ii, data in enumerate(dataloader):
        print(data['gt'].shape)
        print(data['label'].shape)
        # show_tensor_image(data['image'][0])
        # show_tensor_image(data['label'][0])

        # break
