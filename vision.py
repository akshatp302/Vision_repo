import torch 
from torch import nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 



class data_preparation():
    def __init__(self):
        self.data_prepare() 
    
    
    def data_prepare(self):
        
        self.training_data_prepare = torchvision.datasets.CIFAR10(root="data",
                                                             train=True,
                                                             transform=ToTensor(),
                                                             target_transform=None)
        
        self.test_data_prepare = torchvision.datasets.CIFAR10(root="data",
                                                         train=False,
                                                         transform=ToTensor(),
                                                         target_transform=None)
        
        self.class_name_index = self.training_data_prepare.class_to_idx
        
        
        # for i in range(8):
        #     random_image = torch.randint(0,len(self.training_data_prepare),(1,)).item()
        #     image,label = self.training_data_prepare[random_image]
        #     plt.subplot(4,4,i+1)
        #     plt.imshow(image.permute(1,2,0))
        #     plt.title(self.class_name_index[label])
        #     plt.tight_layout()
        #     plt.axis("off")
        # plt.show()
        
        
        
    def data_loader(self):
        self.train_data_ready = DataLoader(dataset=self.training_data_prepare,
                                       batch_size=32,
                                       shuffle=True,
                                       drop_last=False)
        
        self.test_data_ready = DataLoader(dataset=self.test_data_prepare,
                                     batch_size=32,
                                     shuffle=True,
                                     drop_last=True)
        
        
        # image,label = next(iter(self.train_data_ready))
        # for i in range(32):
        #     img = image[i]
        #     plt.subplot(4,8,i+1)
        #     plt.imshow(img.permute(1,2,0))
        #     plt.title(self.class_name_index[label[i]])
        #     plt.tight_layout()
        #     plt.axis("off")
        # plt.show()
        
if __name__ == "__main__":
    
    trail_1 = data_preparation()
    # Image_show = trail_1.data_prepare()
    
