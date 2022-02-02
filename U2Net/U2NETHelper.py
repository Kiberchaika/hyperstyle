from skimage import io
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
from pathlib import Path

from U2Net.data_loader import RescaleT
from U2Net.data_loader import ToTensor
from U2Net.data_loader import ToTensorLab
from U2Net.data_loader import SalObjDataset

from U2Net.model import U2NET 

class U2NETHelper:
    def __init__(self):
        model_dir = './saved_models/u2net/u2net.pth'
        self.net = U2NET(3,1)
        self.net.load_state_dict(torch.load(model_dir))
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()

    # normalize the predicted SOD probability map
    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn

    def get_output(self, image_name, pred):
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np*255).convert('RGB')
        img_name = image_name.split("/")[-1]
        image = io.imread(image_name)
        imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        return imo

    def test(self, path_to_img, path_to_out):
        test_salobj_dataset = SalObjDataset(img_name_list = [path_to_img],
                                            lbl_name_list = [],
                                            transform=transforms.Compose([RescaleT(320),
                                                                        ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)


        for _, data_test in enumerate(test_salobj_dataloader):

            inputs_test = data_test['image']
            print(inputs_test )
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= self.net(inputs_test)

            # normalization
            pred = d1[:,0,:,:]
            pred = self.normPRED(pred)

            out_mask = self.get_output(path_to_img, pred)
            image = Image.open(path_to_img)
            red, green, blue = out_mask.split()
            image.putalpha(red)
            image.save(path_to_out)

            del d1,d2,d3,d4,d5,d6,d7

'''
def main():
    u2net = U2NETHelper()
    u2net.test('03.png', '03_masked.png')
 
if __name__ == "__main__":
    main()
'''