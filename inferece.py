import torch.nn as nn
import torch
from model import UNet3D
from losses import *
from utils import *


def NDISeg(input_numpy_array):
    PATH = r'./model/mobel_level4/model_in_500_dice_mobel4.pth'

    torch.cuda.set_device(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(in_channels=1, out_channels=2, f_maps=4, num_groups=8, num_levels=4, layer_order='bcr', final_sigmoid = False)
    tensor_image = torch.from_numpy(input_numpy_array)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = tensor_image.to(device=device, dtype=torch.float32)

    net.load_state_dict(torch.load(PATH, map_location={"cuda:1":"cuda:0"}))
    net.to(device=device)
    pred = net(tensor_image)   

    criterion = SoftDiceLoss()
    val_lose = criterion(pred, tensor_image_label).item()
    print("inferece Done! loss: ", val_lose) 

    pred = torch.squeeze(pred, 0)
    pred = torch.round(pred)
    pred = pred[1,:,:,:]
    pred = pred.cpu()
    pred = pred.detach().numpy()

    return pred


if __name__ == "__main__":
    
    dest_path = r'/home/hanglok/NDISegData/source/'
    label_path = r'/home/hanglok/NDISegData/label/'
    counter = len(os.listdir(dest_path)) 
    PATH = r'/home/hanglok/NDISphereSegmentationFromCT/model/mobel_level4/model_in_500_dice_mobel4.pth'

    torch.cuda.set_device(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(in_channels=1, out_channels=2, f_maps=4, num_groups=8, num_levels=4, layer_order='bcr', final_sigmoid = False)
    for i in range(counter):
        if i in [49, 10]:
            continue
        image, info = readHMA(dest_path+str(i)+'.mha', down_sample_factor = 2)
        image_label, info_label = readHMA(label_path+str(i)+'.mha', down_sample_factor = 2)
        tensor_image_label = torch.from_numpy(image_label)
        tensor_image_label = torch.unsqueeze(tensor_image_label, 0)
        tensor_image_label = torch.unsqueeze(tensor_image_label, 0)
        tensor_image_label = tensor_image_label.to(device=device, dtype=torch.float32)

        tensor_image = torch.from_numpy(image)
        tensor_image = torch.unsqueeze(tensor_image, 0)
        tensor_image = torch.unsqueeze(tensor_image, 0)
        tensor_image = tensor_image.to(device=device, dtype=torch.float32)
        print("image: ", i)
        
         
        net.load_state_dict(torch.load(PATH, map_location={"cuda:1":"cuda:0"}))
        net.to(device=device)
        pred = net(tensor_image)

        criterion = SoftDiceLoss()
        val_lose = criterion(pred, tensor_image_label).item()
        print("loss: ", val_lose)

        pred = torch.squeeze(pred, 0)
        pred = torch.round(pred)
        pred = pred[1,:,:,:]
        pred = pred.cpu()
        pred = pred.detach().numpy()
        writhMHA_fromNumpy(r'/home/hanglok/NDISegData/result/'+str(i)+'mobel.mha', pred, info)
        del tensor_image_label, tensor_image, pred
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        print()
        print("______________________________________________")
