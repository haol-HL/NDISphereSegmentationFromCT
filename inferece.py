

from losses import *
from utils import *
if __name__ == "__main__":
    import torch.nn as nn
    import torch
    from model import UNet3D
    torch.cuda.set_device(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image, info = readHMA(r'/home/haol/NDISegData/source/2.mha', 2)
    image_label, info_label = readHMA(r'/home/haol/NDISegData/label/2.mha', 2)
    tensor_image_label = torch.from_numpy(image_label)
    tensor_image_label = torch.unsqueeze(tensor_image_label, 0)
    tensor_image_label = torch.unsqueeze(tensor_image_label, 0)
    tensor_image_label = tensor_image_label.to(device=device, dtype=torch.int64)

    tensor_image = torch.from_numpy(image)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = tensor_image.to(device=device, dtype=torch.float32)
    print(tensor_image.size())
    PATH = r'/home/haol/NDISphereSegmentationFromCT/model_in_40_dice_resample0.5.pth'

    
    
    net = UNet3D(in_channels=1, out_channels=2, f_maps=16, num_groups=8, num_levels=5, layer_order='bcr', final_sigmoid = False)
    net.load_state_dict(torch.load(PATH))
    print("load")
    net.to(device=device)
    pred = net(tensor_image)
    # pred = torch.sigmoid(pred)
    criterion = SoftDiceLoss()

    val_lose = criterion(pred, tensor_image_label).item()
    print("loss: ", val_lose)

    # 
    pred = torch.squeeze(pred, 0)
    # pred = torch.round(pred)
    pred = pred[1,:,:,:]
    pred = pred.cpu()
    pred = pred.detach().numpy()
    writhMHA_fromNumpy(r'/home/haol/NDISphereSegmentationFromCT/out02.mha', pred, info)
    print(pred.shape)