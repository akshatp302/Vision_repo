import torch 
from torch import nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 
from timeit import default_timer as timer 
from tqdm import tqdm


class Model_image(nn.Module):
    def __init__(self,input_shape:int,
                    hidden_shape:int,
                    output_shape:int):
        super().__init__()
        
        
        self.flattern = nn.Flatten()
        
        self.Linear_1 = nn.Linear(in_features=input_shape,
                                  out_features= hidden_shape)
        
        self.Linear_2 = nn.Linear(in_features=hidden_shape,
                                  out_features=output_shape
                                  )
        
        self.model = nn.Sequential(self.flattern,
                                   self.Linear_1,
                                   self.Linear_2)
        
    def forward (self,x):
        return self.model(x)
    
    
class Data_prepare():
    
    def __init__(self,batch_size):
        self.batch_size  = batch_size
        self.data_download()
    
    
    def data_download(self):
        
        self.traning_download = datasets.CIFAR10(root="data",
                                                 train=True,
                                                 transform=ToTensor(),
                                                 target_transform=None,
                                                 download=True)
        
        self.test_download = datasets.CIFAR10(root="data",
                                              train=False,
                                              transform=ToTensor(),
                                              download=True,
                                              target_transform=None)
        
        
    def data_loading(self):
        
        
        self.traning_batch_loaded = DataLoader(dataset=self.traning_download,
                                          batch_size = self.batch_size,
                                          shuffle=True,
                                          drop_last= False)
        
        
        self.test_batch_loaded = DataLoader(dataset=self.test_download,
                                       batch_size = self.batch_size,
                                       shuffle= True,
                                       drop_last=True)
        
        return self.traning_batch_loaded, self.test_batch_loaded
        
    
        
        
        
    def visulization_loaded_data(self):
        
        image,label = next(iter(self.traning_loaded))
        label_img = self.traning_download.class_to_idx.items()
        for i in range(32):
            img = image[i]
            plt.subplot(4,8,i+1)
            plt.imshow(img.permute(1,2,0))
            plt.title(label_img[label[i].items()])
            plt.tight_layout()
            plt.axis("off")
        plt.show()


class Trainer:
    def __init__(self,model,data,traning_epoch):
        
        # self.device = torch.device("cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data = data
       
        
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.01)
       
        self.traning_epoch = traning_epoch
        
        
        
        
        # self.traning_loop()
        # self.evulation()
        # self.accuracy_check()
    
       
    # def progress_bar(self):
    #     for i in tqdm(range(self.epoch)):
    #         print(i)
       
       
    def accuracy_check(self,Y_labels,Model_output):
        # def accuracy_check(self,Y_labels,Model_output):
        correct = torch.eq(Y_labels,Model_output).sum().item()
        accuracy= (correct/len(Model_output))*100
        return accuracy   
    
    def traning_loop(self):
    
        torch.manual_seed(42)
        # self.train_loss = 0
        self.traning_time_start = timer()
        
        traning_batch_loaded , _ = self.data.data_loading()
        
        for epoch_i in tqdm(range(self.traning_epoch)):
            print(f" Epoch_no: {epoch_i}")
            
            initial_loss = 0
            
            for batch,(Actual_data_X_train,Actual_label_Y_train)in enumerate(traning_batch_loaded):
                self.model.train()
                
                traning_model_prediction_Y = self.model(Actual_data_X_train)
                
                train_loss = self.loss_function(traning_model_prediction_Y, Actual_label_Y_train)
                initial_loss += train_loss.item()
                
                self.optimizer.zero_grad()
                
                train_loss.backward()
                self.optimizer.step()
            
                if batch % 100 ==0:
                    print(f"Looked at {batch*len(Actual_data_X_train)}/{len(traning_batch_loaded)} samples ")
            
            
            self.train_loss = initial_loss/len(traning_batch_loaded)
            # self.train_loss /= len(traning_batch_loaded)
            
        self.traning_time_stop = timer()
        
    
    
    def evulation(self):
        initial_test_loss = 0
        initial_test_accuracy = 0
        self.model.eval()
        
        self.test_time_start = timer()
        
        _ , self.test_batch_loaded = self.data.data_loading()
        
        with torch.inference_mode():
            
            for Actual_data_X_test ,Actual_label_Y_test in self.test_batch_loaded:
                
                testing_model_prediction_Y = self.model(Actual_data_X_test)
                
                test_loss = self.loss_function(testing_model_prediction_Y,Actual_label_Y_test)
               
                initial_test_loss += test_loss.item()

                test_accuracy = self.accuracy_check(Y_labels=Actual_label_Y_test,
                                                    Model_output=testing_model_prediction_Y.argmax(dim =1))

                initial_test_accuracy += test_accuracy
        # Test_loss_average 
        
            self.test_loss = initial_test_loss/len(self.test_batch_loaded)   
        
            # self.initial_test_loss/= len(test_batch_loaded)
            
        #Test_loss and accuracy 
        
            self.test_accuracy= initial_test_accuracy/ len(self.test_batch_loaded)
            
            
        
        self.test_time_stop = timer()
    
        print(f"Train loss:{self.train_loss:.3f} | Test loss :{ self.test_loss:.2f} and | Test_accuracy {self.test_accuracy} at epoch {self.traning_epoch:.2f}  ")

        timing_print_train = self.timing(start_time=self.traning_time_start,
                                   end_time=self.traning_time_stop)
        
        timing_print_test = self.timing(start_time = self.test_time_start,
                                        end_time = self.test_time_stop)
        
        print(f"The model time during traning {timing_print_train} and model time for evulation {timing_print_test}")
        
    
    
    # @staticmethod
    def timing(self, start_time,end_time):
        # start_time = timer()
        # end_time = timer()
        print(f"Total time = {(end_time-start_time)/60:.2f} on device {self.device} seconds ")
        
        
    def eval_model(self):
        
        loss = 0
        accuracy =0
        
        # _test_data_loaded = self.data.data_loading()
        _,test_batch_loaded = self.data.data_loading()
        
        self.model.eval()
        with torch.inference_mode():
            for X,Y in test_batch_loaded:
                model_prediction_y = self.model(X)
                
                loss += self.loss_function(model_prediction_y,Y)
                accuracy += self.accuracy_check(Y_labels=Y,
                                                Model_output=model_prediction_y.argmax(dim=1))
                
                loss /= len(self.data.data_loading())
                accuracy /= len(self.data.data_loading())
                
        return {"MOdel_name = " :self.model.__class__.__name__,"model_ NAme" :loss.item(),
                "model_accuracy":accuracy}
            
        
    
    
    
    
    

trail_x = Trainer(model=Model_image(input_shape=3*32*32,
                                    hidden_shape=20,
                                    output_shape=10),
                  
                  data= Data_prepare(batch_size=32),
                  
                  traning_epoch=1)

print(trail_x.eval_model())

print(torch.cuda.is_available())
print(torch.version.cuda )
