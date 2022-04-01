from logging import critical
from model import UNet3D
from DataUtils import My_dataset
from torch import optim
from losses import DiceLoss
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
 
class trainer():
    def __init__(self, model, optimizer, loss, train_loader, val_loader, device, epoches, lr) -> None:
        self.net = model
        self.optimizer = optimizer
        self.criterion = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.maxepoches = epoches
        self.lr = lr

    def train(self):
        # best_loss统计，初始化为正无穷
        best_loss = float('inf')
        # 训练epochs次
        print("begin training")
        for epoch in range(self.maxepoches):
            print("epoch: ", epoch)
            # 训练模式
            net.train()
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
                print('Loss/train', loss.item())
                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model.pth')
                # 更新参数
                print('\r', "training...  {:.2f}%, best loss in this epoch: {}".format(i_batch / len(self.train_loader) * 100, best_loss), end=' ', flush=True)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                val_lose = float('inf')
                for i_batch, batch_data in enumerate(self.train_loader):
                    image = batch_data[0]
                    label = batch_data[1]

                    image = torch.unsqueeze(image, 0)
                    label = torch.unsqueeze(label, 0)

                    image = image.to(device=self.device, dtype=torch.float32)
                    label = label.to(device=self.device, dtype=torch.float32)

                    pred = net(image)
                    # 计算loss
                    val_lose += self.criterion(pred, label).item()


                print("epoch: {} done".format(epoch))
                print("best train loss: {}".format(best_loss)) 
                print("validation loss: {}".format(val_lose))
                print()
                print("#"*50)
                print()

    def fit(self):
        self.train()

 
 
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(in_channels=1, out_channels=1, f_maps=4, num_groups=4, num_levels=3)
    net.to(device=device)

    dataset = My_dataset(r'D:\TPS\SepheresSegData\source', r'D:\TPS\SepheresSegData\label')
    
    train_num = int(0.8 * len(dataset))
    val_num = len(dataset) - train_num
    train_set, val_set = torch.utils.data.random_split(dataset, [train_num, val_num])
    train_data_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    val_data_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    print("load data completed")

    optimizer = optim.RMSprop(net.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    criterion = DiceLoss()
    
    Unet_trainer = trainer(net, optimizer, criterion, train_data_loader, val_data_loader, device, 200, 0.0001)
    Unet_trainer.fit()