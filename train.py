
from model import UNet3D
from DataUtils import My_dataset
from torch import optim
from losses import DiceLoss
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import os
from Haologger import *
import time

class trainer():
    def __init__(self, model, optimizer, lr_scheduler, loss, train_loader, val_loader, device, epoches, lr) -> None:
        self.net = model
        self.optimizer = optimizer
        self.criterion = loss
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.maxepoches = epoches
        self.lr = lr
        

    def train(self, logger):
        # best_loss统计，初始化为正无穷
        best_loss = float('inf')
        # 训练epochs次
        print("begin training")
        if logger is not None:
            logger.info("Model: {}".format(self.net))
            logger.info("optimizer: {}".format(self.optimizer))
            logger.info("loss: {}".format(self.criterion))
            logger.info("#"*20)
            logger.info(" "*20)
            logger.info("------------begin training--------------")
        for epoch in range(self.maxepoches):
            
            time_begin = time.time()
            print("epoch: ", epoch)
            if logger is not None:
                logger.info("epoch: {}".format(epoch))
            
            # 按照batch_size开始训练
            for i_batch, batch_data in enumerate(self.train_loader):

                image = batch_data[0]
                label = batch_data[1]
                image = torch.unsqueeze(image, 0)
                label = torch.unsqueeze(label, 0)

                self.optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                loss = self.criterion(pred, label)
                # print('Loss/train', loss.item())
                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model.pth')
                # 更新参数
                # print('\r', "training...  {:.2f}%, best loss in this epoch: {}".format(i_batch / len(self.train_loader) * 100, best_loss), end=' ', flush=True)
                if logger is not None:
                    logger.info("epoch: {}, lr: {}, {:.2f}%, loss: {}".format(epoch, self.lr_scheduler.get_lr(), (i_batch  + 1) / len(self.train_loader) * 100, loss))
                loss.backward()
                
                
                del image, label, pred, loss
                torch.cuda.empty_cache()
                self.optimizer.step()

            self.lr_scheduler.step()
            with torch.no_grad():
                val_lose = 0
                for i_batch, batch_data in enumerate(self.train_loader):
                    image = batch_data[0]
                    label = batch_data[1]

                    image = torch.unsqueeze(image, 1)
                    label = torch.unsqueeze(label, 1)

                    image = image.to(device=self.device, dtype=torch.float32)
                    label = label.to(device=self.device, dtype=torch.float32)

                    pred = net(image)
                    # 计算loss
                    val_lose += self.criterion(pred, label).item()
                val_lose /= i_batch

                # print("epoch: {} done".format(epoch))
                # print("best train loss: {}".format(best_loss)) 
                # print("validation loss: {}".format(val_lose))
                # print()
                # print("#"*50)
                # print()
                
                time_end = time.time()
                if logger is not None:
                    logger.info("epoch: {} done".format(epoch))
                    logger.info("best train loss: {}".format(best_loss)) 
                    logger.info("validation loss: {}".format(val_lose))
                    logger.info("time usage in this epoch: {:2f}s".format(time_end - time_begin))
                    logger.info("#"*50)
                    logger.info(" "*50)

    def fit(self, logger):
        self.train(logger)

 
 
if __name__ == "__main__":
    torch.cuda.set_device(1)
    log_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "log")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log = HagLogger(os.path.join(log_path, "log_mod.log"))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(in_channels=1, out_channels=1, f_maps=4, num_groups=4, num_levels=2, layer_order='bcr')
    net.to(device=device)

    dataset = My_dataset(r'/home/haol/NDISpheresData/source', r'/home/haol/NDISpheresData/label')
    
    train_num = int(0.8 * len(dataset))
    val_num = len(dataset) - train_num
    train_set, val_set = torch.utils.data.random_split(dataset, [train_num, val_num])
    train_data_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    val_data_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    print("load data completed")

    optimizer = optim.Adam(net.parameters(),
                            lr=0.001,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=0,
                            amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1, last_epoch=-1)
    criterion = nn.BCEWithLogitsLoss()
    
    Unet_trainer = trainer(net, optimizer, scheduler, criterion, train_data_loader, val_data_loader, device, 100, 0.0001)
    Unet_trainer.fit(log)
