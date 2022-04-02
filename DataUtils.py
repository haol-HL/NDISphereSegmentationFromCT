import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils import *
import os

class My_dataset(Dataset):
    def __init__(self, src_image_path, trg_image_path):
        super().__init__()
        # assert device in ["gpu", 'cpu']
        # if device == 'gpu':
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     print("cant use gpu")
        # else:
        #     self.device = torch.device("cpu")
        self.src,  self.trg = [], []
        
        src_filenames=os.listdir(src_image_path)
        trg_filenames=os.listdir(trg_image_path)
        assert len(trg_image_path) != len(src_filenames), "trg size didnt match src size trg:{} != src:{}".format(len(trg_filenames), len(src_filenames))
        for i in range(len(src_filenames)):
            print('\r', "loading data: {} / {}".format(i + 1, len(src_filenames)), end=' ', flush=True)
            src_image, _ = readHMA(os.path.join(src_image_path, src_filenames[i]))
            trg_image, _ = readHMA(os.path.join(trg_image_path, trg_filenames[i]))

            src_tensor_image = torch.from_numpy(src_image).cpu() 
            trg_tensor_image = torch.from_numpy(trg_image).cpu()

            self.src.append(src_tensor_image)
            self.trg.append(trg_tensor_image)
           
    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src) 


if __name__ == '__main__':
    mydataset = My_dataset(r'D:\TPS\SepheresSegData\source', r'D:\TPS\SepheresSegData\label')
    data_loader = DataLoader(mydataset, batch_size=1, shuffle=False)
    for i_batch, batch_data in enumerate(data_loader):
        print(i_batch, len(data_loader))  # 打印batch编号
        print(batch_data[0].size())  # 打印该batch里面src
        print(batch_data[1].size())  # 打印该batch里面trg

