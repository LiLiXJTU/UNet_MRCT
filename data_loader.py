import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data


def unnor(data):
    data = data * np.std(data)
    data += np.mean(data)
    return data


def get_data_path():
    path_all = r'D:\data\MR_CT\brain'
    #path_all = '/mnt/sda/li/MR_CT/brain'
    train_files, val_files, test_files = [], [], []

    for j, file in enumerate(sorted(os.listdir(path_all))):
        for i,name in enumerate(os.listdir(path_all + '/' + file + '/CT')):
            if file=='test':
                test_files.append(path_all + '/' + file + '/MR/' + name)
            elif file=='val':
                val_files.append(path_all + '/' + file + '/MR/' + name)
            elif file == 'train':
                train_files.append(path_all + '/' + file + '/MR/' + name)
    return train_files, val_files, test_files



class Dataset(data.Dataset):
    def __init__(self, imgs, shape=(256, 256), return_roi=False):
        self.roi = return_roi
        self.imgs = imgs
        self.input_shape = shape

    def predict(self, img_path, model, device):
        cbct, pre = self.read_data(img_path)
        with torch.no_grad():
            cbct = cbct.to(device)[None]
            pre_ct = model(cbct)
        pre_ct = pre_ct.cpu().numpy()[0, 0]
        pre_ct = unnor(pre_ct) * .7 + unnor(pre[0].numpy()) * .3
        return pre_ct

    def read_data(self, img_path):
        label_path = img_path.replace('/MR/', '/CT/')
        mr = Image.fromarray(np.load(img_path))
        ct = Image.fromarray(np.load(label_path))
        mr = mr.resize((self.input_shape[0], self.input_shape[1]), Image.BICUBIC)
        ct = ct.resize((self.input_shape[0], self.input_shape[1]), Image.BICUBIC)

        if self.roi:
            roi = np.array(ct, np.float64)
        mr = np.array(mr, np.float64) - (np.mean(np.array(mr, np.float64)))
        mr = mr / (np.std(np.array(mr, np.float64)))
        mr = mr[..., None]

        ct_mean = np.mean(np.array(ct, np.float64))
        ct_std = np.std(np.array(ct, np.float64))
        ct = np.array(ct, np.float64) - (np.mean(np.array(ct, np.float64)))
        ct = ct / (np.std(np.array(ct, np.float64)))
        ct = ct[..., None]
        # if self.use_transform:
        #     transformed = self.transform(image=mr, mask=ct)
        #     mr = transformed['image']
        #     ct = transformed['mask']

        mr = np.transpose(mr, [2, 0, 1])
        ct = np.transpose(ct, [2, 0, 1])

        mr = torch.from_numpy(mr).type(torch.FloatTensor)
        ct = torch.from_numpy(ct).type(torch.FloatTensor)
        # print(jpg.shape)
        if self.roi:
            return mr, ct, roi, ct_mean,ct_std
        else:
            return mr, ct, ct_mean,ct_std

    def __getitem__(self, index):
        if self.roi:
            img_x, img_y, roi_x, ct_mean,ct_std = self.read_data(self.imgs[index])
            # print(1)
            # print('ct_mean',ct_mean)
            # print('ct_std', ct_std)
            return img_x, img_y, roi_x, ct_mean,ct_std
        else:
            img_x, img_y, ct_mean,ct_std = self.read_data(self.imgs[index])
            # print(2)
            # print('ct_mean',ct_mean)
            # print('ct_std', ct_std)
            return img_x, img_y, ct_mean,ct_std

    def __len__(self):
        return len(self.imgs)
