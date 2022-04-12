
from model import UNet3D
from DataUtils import My_dataset
from torch import optim
from losses import *
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import os
from Haologger import *
import time
def binary_acc(preds, y):
    p = preds.view(2, -1).T
    p = p.gather(1,y.view(-1,1))
    correct = torch.round(p).sum().item()
    acc = correct / len(p)
    return acc

def FalsePos_rate(preds, y):
    p = preds.view(2, -1).T
    y = y.view(-1)
    
    pos = torch.round(p)[:,1]
    pos_num = torch.count_nonzero(pos).item()
    if pos_num == 0:
        rate = -1
    else:
        FalsePos_num = torch.count_nonzero((y==0) & (pos==1)).item()
        rate = FalsePos_num / pos_num
    return rate

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
        train_loss = 0
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
            counter = 0
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

                image = image.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.int64)

                pred = net(image)

                loss = self.criterion(pred, label)
                train_loss += loss

                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model_dice_resample0.5.pth')
                # 更新参数
                print('\r', "epoch: {}, lr: {}, {:.2f}%, loss: {}, counter:{}".format(epoch, self.lr_scheduler.get_lr(), (i_batch  + 1) / len(self.train_loader) * 100, loss, counter, end=' ', flush=True))
                if logger is not None:
                    logger.info("epoch: {}, lr: {}, {:.2f}%, loss: {}, counter:{}".format(epoch, self.lr_scheduler.get_lr(), (i_batch  + 1) / len(self.train_loader) * 100, loss, counter))
                loss.backward()
                
                self.optimizer.step()
                del image, label, pred, loss
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                
                counter += 1

            self.lr_scheduler.step()
            train_loss /= counter
            with torch.no_grad():
                val_lose = 0
                acc_all = 0
                FlasePos_rate = 0
                counter = 0
                for i_batch, batch_data in enumerate(self.val_loader):
                    image = batch_data[0]
                    label = batch_data[1]

                    image = torch.unsqueeze(image, 0)
                    label = torch.unsqueeze(label, 0)

                    image = image.to(device=self.device, dtype=torch.float32)
                    label = label.to(device=self.device, dtype=torch.int64)

                    pred = net(image)
                    # 计算loss
                    val_lose += self.criterion(pred, label).item()
                    acc_all += binary_acc(pred, label)
                    FlasePos_rate += FalsePos_rate(pred, label)
                    
                
                    counter += 1
                val_lose /= counter
                acc_all /= counter
                FlasePos_rate /= counter

                time_end = time.time()
                


                print("epoch: {} done".format(epoch))
                print("train train loss: {}".format(train_loss)) 
                print("validation loss: {}".format(val_lose))
                print("Mean validation ACC: {}".format(acc_all))
                print("Mean Flase postive rate: {}".format(FlasePos_rate))
                print("time usage in this epoch: {:2f}s".format(time_end - time_begin))
                print("#"*50)
                print(" "*50)
                if logger is not None:
                    logger.info("epoch: {} done".format(epoch))
                    logger.info("train train loss: {}".format(train_loss)) 
                    logger.info("validation loss: {}".format(val_lose))
                    logger.info("Mean validation ACC: {}".format(acc_all))
                    logger.info("Mean Flase postive rate: {}".format(FlasePos_rate))
                    logger.info("time usage in this epoch: {:2f}s".format(time_end - time_begin))
                    logger.info("#"*50)
                    logger.info(" "*50)

            if (epoch + 1) % (int(self.maxepoches / 5)) == 0:
                torch.save(net.state_dict(), 'model_in_{}_dice_resample0.5.pth'.format(epoch+1))

    def fit(self, logger):
        self.train(logger)

 
 
if __name__ == "__main__":
    torch.cuda.set_device(0)
    log_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "log")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log = HagLogger(os.path.join(log_path, "log_dice_resample0.5.log"))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(in_channels=1, out_channels=2, f_maps=16, num_groups=8, num_levels=5, layer_order='bcr', final_sigmoid = False)
    net.to(device=device)

    dataset = My_dataset(r'/home/haol/NDISegData/source', r'/home/haol/NDISegData/label', 0.5)
    
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
    criterion = SoftDiceLoss()
    
    Unet_trainer = trainer(net, optimizer, scheduler, criterion, train_data_loader, val_data_loader, device, 100, 0.0001)
    Unet_trainer.fit(log)
