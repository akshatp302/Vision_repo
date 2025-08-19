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
  
    def __init__(self):
 
        super().__init__()
        
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1),
            )

        conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      stride=1,
                      kernel_size=2,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      stride=1,
                      kernel_size=2,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        flatern = nn.Flatten()
        Linear_1 = nn.Linear(in_features=128*7*7,out_features=1200)
        non_linear_1 = nn.ReLU()
        
        Linear_2 = nn.Linear(in_features=1200,out_features=300) 
        non_linear_2 = nn.ReLU()    
        
        Linear_3 = nn.Linear(in_features=300,out_features=40)
        non_linear_3 = nn.ReLU()    
        
        Linear_4 = nn.Linear(in_features=40,out_features=10)                
        
        self.model = nn.Sequential(
            self.conv_1,
            conv_2,
            conv_3,
            flatern,
            Linear_1,
            non_linear_1,
            Linear_2,
            non_linear_2,
            Linear_3,
            non_linear_3,
            Linear_4
        )
      

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
        
        self.device = torch.device("cuda")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data = data
       
        
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=1e-3)
       
        self.traning_epoch = traning_epoch
        
        
        
        
        self.traning_loop()
        self.evulation()
     

       
    def accuracy_check(self,Y_labels,Model_output):
        # def accuracy_check(self,Y_labels,Model_output):
        correct = torch.eq(Y_labels,Model_output).sum().item()
        accuracy= (correct/len(Model_output))*100
        return accuracy   


    def curves(self,x_axis,y_axis,x_label,y_label):
        plt.figure(figsize=(10,5))
        plt.plot(x_axis,y_axis,)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        # plt.savefig(f"loss_curve_epoch_{self.traning_epoch}.png", dpi=300)

        plt.show()
        # plt.close() 
    
    
    
    def traning_loop(self):

        self.train_loss_storage = []
        # to_numpy  = ToTensor.to_numpy

        torch.manual_seed(42)
        # self.train_loss = 0
        self.traning_time_start = timer()
        
        traning_batch_loaded , _ = self.data.data_loading()
        
        for epoch_i in tqdm(range(self.traning_epoch)):
            print(f" Epoch_no: {epoch_i}")
            
            initial_loss = 0
            
            for batch,(Actual_data_X_train,Actual_label_Y_train)in enumerate(traning_batch_loaded):
                self.model.train()
                
                Actual_data_X_train = Actual_data_X_train.to(self.device,non_blocking=True)
                Actual_label_Y_train = Actual_label_Y_train.to(self.device,non_blocking=True)
            
                
                traning_model_prediction_Y = self.model(Actual_data_X_train)
                
                train_loss = self.loss_function(traning_model_prediction_Y, Actual_label_Y_train)
                initial_loss += train_loss.item()
                
                self.optimizer.zero_grad()
                
                train_loss.backward()
                self.optimizer.step()
            
                if batch % 100 ==0:
                    print(f"Looked at {batch*len(Actual_data_X_train)}/{len(traning_batch_loaded)} samples ")

            print(f"{train_loss:.2f} at batch {batch}")

            self.train_loss_storage.append(train_loss.item())

            self.train_loss = initial_loss/len(traning_batch_loaded)
            # self.train_loss /= len(traning_batch_loaded)
            
        torch.save(self.model.state_dict(),f"Cifar_10.pth")
        print("Model_params saved successfully.")

        self.traning_time_stop = timer()
        
    
    
    def evulation(self):
        initial_test_loss = 0
        initial_test_accuracy = 0
        self.model.eval()
        
        self.test_time_start = timer()
        
        _ , self.test_batch_loaded = self.data.data_loading()
        
        with torch.inference_mode():
            
            for Actual_data_X_test ,Actual_label_Y_test in self.test_batch_loaded:
                
                Actual_data_X_test = Actual_data_X_test.to(self.device,non_blocking=True)   
                Actual_label_Y_test = Actual_label_Y_test.to(self.device,non_blocking=True) 
                
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
       


        epoch_graph = list(range(1,self.traning_epoch+1))
        self.curves(x_axis=epoch_graph,y_axis=self.train_loss_storage,x_label="Epochs",y_label="Train Loss")

    def new_method(self):
        print(self.shape_check.shape)


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
                
                
                X = X.to(self.device,non_blocking=True)
                Y = Y.to(self.device,non_blocking=True)
                model_prediction_y = self.model(X)
                
                loss += self.loss_function(model_prediction_y,Y)
                accuracy += self.accuracy_check(Y_labels=Y,
                                                Model_output=model_prediction_y.argmax(dim=1))
                
            loss /= len(self.data.data_loading())
            accuracy /= len(self.data.data_loading())
                
        return {"MOdel_name = " :self.model.__class__.__name__,"model_ NAme" :loss.item(),
                "model_accuracy":accuracy}
            
            

    
    
    
    
    



# trail_y = Trainer(model=Model_image(),
#                   data=Data_prepare(batch_size=64),
#                   traning_epoch=4)
# print(trail_y.eval_model())
# print(trail_x.eval_model())

# print(torch.cuda.is_available())
# print(torch.version.cuda)



params = torch.load("Cifar_10.pth",weights_only=True)
trail_z = Model_image() 
trail_z.load_state_dict(params)
trail_z.eval()


torch.random.manual_seed(42)

data_prep = Data_prepare(batch_size=64)
_, test_batch_loaded = data_prep.data_loading()

classes_names = data_prep.test_download.classes
image,label = next(iter(test_batch_loaded))
image, label = image.to(trail_z.model[0][0].weight.device), label.to(trail_z.model[0][0].weight.device) 

with torch.inference_mode():
    raw_output = trail_z.model(image)
    probability = raw_output.softmax(dim=1) 
    predictions = probability.argmax(dim=1) 
    
 
plt.figure(figsize=(12, 5))
for i in range(15):
    img = image[i].cpu().permute(1, 2, 0)   # CHW -> HWC for plotting
    true_label = classes_names[label[i].item()]
    pred_label = classes_names[predictions[i].item()]
    conf = probability[i][predictions[i]].item()

    plt.subplot(3, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}\n({conf*100:.1f}%)", fontsize=9)

plt.tight_layout()
plt.show()
print(classes_names)