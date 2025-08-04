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
        self.data_loader()
    
    
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
        print(self.class_name_index)
        
        
        for i in range(20):
            random_image = torch.randint(0,len(self.training_data_prepare),(1,)).item()
            image,label = self.training_data_prepare[random_image]
            # if i ==1:
            #     plt.figure(figsize=(5,10))
            plt.subplot(10,10,i+1)
            plt.imshow(image.permute(1,2,0))
            # plt.title(self.class_name_index[label])
            
            plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        
        
    def data_loader(self):
        self.train_data_ready = DataLoader(dataset=self.training_data_prepare,
                                       batch_size=32,
                                       shuffle=True,
                                       drop_last=False)
        
        self.test_data_ready = DataLoader(dataset=self.test_data_prepare,
                                     batch_size=32,
                                     shuffle=True,
                                     drop_last=True)
        
        
        image,label = next(iter(self.train_data_ready))
        for i in range(32):
            img = image[i]
            plt.subplot(4,8,i+1)
            plt.imshow(img.permute(1,2,0))
            # plt.title(self.class_name_index[label[i]])
            plt.tight_layout()
            plt.axis("off")
        plt.show()



class image_model(nn.Module):
    def __init__(self,input_shape:int,
                 hidden_shape:int,
                 output_shape:int):
        # self.data_calling = data_preparation()
        super().__init__()
        
    
        self.flattern = nn.Flatten()
        
        self.linear_1 = nn.Linear(in_features =input_shape,
                                  out_features=hidden_shape)
        
        self.linear_2 = nn.Linear(in_features=hidden_shape,
                                  out_features=output_shape)
        
        
        self.sequential = nn.Sequential(self.flattern,
                                        self.linear_1,
                                        self.linear_2,
                                        )
        
        
    def forward(self,x):
        return self.sequential(x)
        
    


if __name__ == "__main__":
    
    
    names = data_preparation()
    names_extraction = names.class_name_index
    
    trail_2 = image_model(input_shape=28*29,
                          hidden_shape=10,
                          output_shape=len(names_extraction))
    no_paramaters = trail_2.state_dict().items()
    print(len(no_paramaters))
    
    for names,param in no_paramaters:
        print(f"the name is {names} and the values is {param}")

    print(trail_2)
    
