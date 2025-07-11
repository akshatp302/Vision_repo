import torch 
from torch import nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 



class example (nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

class data:
    def load_data(self):
        self.Traning_data_set =torchvision.datasets.CIFAR10(root="data",
                                                            train=True,
                                                        transform=ToTensor(),
                                                        download=True,
                                                        target_transform=None)
        
        self.Test_data_set = torchvision.datasets.CIFAR10(root="data",
                                                          train=False,
                                                          download=True,
                                                          transform=ToTensor(),
                                                          target_transform=None)
        # class_names = self.Traning_data_set.class_to_idx
        self.class_name = self.Traning_data_set.classes
        classes_index = self.Traning_data_set.class_to_idx
        print(classes_index)
        return self.Traning_data_set
        
    def visulization(self,row,column):
        self.no_row = row
        self.no_column = column
        self.load_data()
        plt.figure(figsize=(10,10))
       
        
        for i in range(1,self.no_column*self.no_column+1):
            image,label = self.Traning_data_set[torch.randint(0,len(self.Traning_data_set),(1,)).item()]
            plt.subplot(row,column,i)
             
            # plt.imshow(image.squeeze())
            plt.imshow(image.permute(1, 2, 0)) 
            
            
        plt.show()
        
