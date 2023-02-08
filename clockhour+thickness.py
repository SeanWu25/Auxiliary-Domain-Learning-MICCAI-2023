#!/usr/bin/env python
# coding: utf-8

# In[44]:


import cv2
import numpy as np
import torchvision.transforms as transforms
import time
import random
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
import os
import pandas as pd 
import numpy as np   
import cv2          
import matplotlib.pyplot as plt  
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models, utils
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from skimage import io, transform
from torch.optim import lr_scheduler
from skimage.transform import AffineTransform, warp
from tqdm import tqdm
device = torch.device('cuda')


# In[45]:


import albumentations as A
from albumentations.pytorch.transforms import ToTensor

train_transform = A.Compose([
    
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.20, rotate_limit=90, p=1),
   
])


# In[46]:


class Glauc(Dataset):
  def __init__(self,csv_file,transform = None,transform2 = None):
    self.annotations = pd.read_csv(csv_file)
    
    self.transform = transform
    self.transform2 = transform2
  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,index):
    label = torch.tensor(int(self.annotations.iloc[index,2]))
    age = torch.tensor((self.annotations.iloc[index,7]))
    md = torch.tensor((self.annotations.iloc[index,6]))
    c1 =  torch.tensor(int(self.annotations.iloc[index,8]))
    c2 =  torch.tensor(int(self.annotations.iloc[index,9]))
    c3 =  torch.tensor(int(self.annotations.iloc[index,10]))
    c4 =  torch.tensor(int(self.annotations.iloc[index,11]))
    c5 =  torch.tensor(int(self.annotations.iloc[index,12]))
    c6 =  torch.tensor(int(self.annotations.iloc[index,13]))
    c7 =  torch.tensor(int(self.annotations.iloc[index,14]))
    c8 =  torch.tensor(int(self.annotations.iloc[index,15]))
    c9 =  torch.tensor(int(self.annotations.iloc[index,16]))
    c10 =  torch.tensor(int(self.annotations.iloc[index,17]))
    c11 =  torch.tensor(int(self.annotations.iloc[index,18]))
    c12 =  torch.tensor(int(self.annotations.iloc[index,19]))
    thickness = torch.tensor(int(self.annotations.iloc[index,20]))
    
    try:
        m1 = torch.tensor(int(self.annotations.iloc[index,21]))
        m2 = torch.tensor(int(self.annotations.iloc[index,22]))
        m3 = torch.tensor(int(self.annotations.iloc[index,23]))
        m4 = torch.tensor(int(self.annotations.iloc[index,24]))
        m5 = torch.tensor(int(self.annotations.iloc[index,25]))
        m6 = torch.tensor(int(self.annotations.iloc[index,26]))
        m7 = torch.tensor(int(self.annotations.iloc[index,27]))
        m8 = torch.tensor(int(self.annotations.iloc[index,28]))
    except:
        m1 = m2 = m3=m4=m5=m6=m7=m8 =torch.tensor(0)
        
       
    

    

    image = cv2.imread(self.annotations.iloc[index,3])
    image1 = cv2.imread(self.annotations.iloc[index,4])
    image2 = cv2.imread(self.annotations.iloc[index,5])
    if image1 is None:
        image1 = 255 * np.ones(shape=[224, 224, 3], dtype=np.uint8)
    if image2 is None:
        image2 = 255 * np.ones(shape=[224, 224, 3], dtype=np.uint8)
    
 
 
    str1 = self.annotations.iloc[index,3]

    try:
        segmOne = str1.split("OD\\")
        segmTwo = segmOne[1].split(".jpg")
        segmThree = segmTwo[0]    
        segmThree += ".png"  
                
    except:
        segmOne = str1.split("OS\\")
        segmTwo = segmOne[1].split(".jpg")
        segmThree = segmTwo[0]
        segmThree += ".png"
        
        image = np.fliplr(image).copy()
        if image1 is not None:
            image1 = np.fliplr(image1).copy()
        if image2 is not None:
            image2 = np.fliplr(image2).copy()

    
    if self.transform is not None:
        augmentations = self.transform(image=image)
        image = augmentations["image"]
        
        if image1 is not None:
        
            augmentations1 = self.transform(image=image1)
            image1 = augmentations1["image"]
        if image2 is not None:
        
            augmentations2 = self.transform(image=image2)
            image2 = augmentations2["image"]
    image = self.transform2(image)
    if image1 is not None:
        image1 = self.transform2(image1)
    if image2 is not None:
        image2 = self.transform2(image2)
    



                    
    return (image,image1,image2,label,age,md,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,thickness,m1,m2,m3,m4,m5,m6,m7,m8)


# In[47]:


dataset = Glauc(csv_file = 'C:/Users/Admin/Desktop/SeanMaculaProject/train.csv',transform = train_transform,transform2 = transforms.ToTensor())
dataset1 = Glauc(csv_file ='C:/Users/Admin/Desktop/SeanMaculaProject/val.csv',transform2 = transforms.ToTensor())
dataset2 = Glauc(csv_file = 'C:/Users/Admin/Desktop/SeanMaculaProject/test.csv',transform2 = transforms.ToTensor())
batch_size = 8
train_loader = DataLoader(dataset =  dataset, batch_size = 16,shuffle = True)
val_loader = DataLoader(dataset =  dataset2, batch_size =16,shuffle = False)
test_loader =DataLoader(dataset =  dataset1, batch_size = 1,shuffle = False)


# In[48]:



import torchvision.models
criterion = nn.CrossEntropyLoss()



class AuxOut(nn.Module):
    def __init__(self, pretrained):
        super(AuxOut, self).__init__()
    
        if pretrained is True:
           # self.model = models.efficientnet_b4(pretrained = True)


            self.model = models.resnet18(pretrained=True)
          #  for param in self.model.parameters():
              #  param.requires_grad = False
            self.model.fc = nn.Linear(512, 512)
            self.model = self.model.to(device)
        else:
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1024)
            self.model = self.model.to(device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1536,1) #main progression task
        
        self.clockfc = nn.Linear(1536,13)
        
        
        self.linear = nn.Linear(23,50)
        self.linear1 = nn.Linear(50,100)
        self.linear2 = nn.Linear(100,100)
        

    
 


     



    

        
    def forward(self,image,image1,image2,label,age,md,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,thickness,m1,m2,m3,m4,m5,m6,m7,m8):
      

        
      #  features = self.relu(features)
        
       
        y = self.model(image) #feature extraction first imag
        z = self.model(image1)
        fin = self.model(image2)
       # y = self.bn(y)
       # y = self.relu(y)

       


        combined = torch.cat((y,z,fin),dim=1) #concat features
       # combined = torch.cat((y,age,md,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,thickness),dim=1) #concat features
       # combined = self.dropout1(combined)
       
       # out = self.fc2(combined) #main task
        out = self.fc2(combined)
        thickness = self.clockfc(combined)
        
        return{'label1' : out,'label2':thickness}

finalmodel = AuxOut(True)

print(finalmodel)


# In[49]:


import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import auc,precision_recall_curve,plot_roc_curve, accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,roc_auc_score


# In[50]:


import os
import time
from glob import glob
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
def train2(model, loader, optimizer, loss_fn,mse, device):
    epoch_loss = 0.0

    model.train()
    for image,image1,image2,label,age,md,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,thickness,m1,m2,m3,m4,m5,m6,m7,m8 in loader:
        x1 = image.to(device)
        x2 = image1.to(device)
        x3 = image2.to(device)
        x4 = label.to(device).unsqueeze(axis =1)
        x5 = age.to(device).unsqueeze(axis =1)
        x6 = md.to(device).unsqueeze(axis =1)
        x7 = c1.to(device).unsqueeze(axis =1)
        x8 = c2.to(device).unsqueeze(axis =1)
        x9 = c3.to(device).unsqueeze(axis =1)
        x10 = c4.to(device).unsqueeze(axis =1)
        x11 = c5.to(device).unsqueeze(axis =1)
        x12 = c6.to(device).unsqueeze(axis =1)
        x13 = c7.to(device).unsqueeze(axis =1)
        x14 = c8.to(device).unsqueeze(axis =1)
        x15 = c9.to(device).unsqueeze(axis =1)
        x16 = c10.to(device).unsqueeze(axis =1)
        x17 = c11.to(device).unsqueeze(axis =1)
        x18 = c12.to(device).unsqueeze(axis =1)
        x19 = thickness.to(device).unsqueeze(axis =1)
        optimizer.zero_grad()
        x6 = x6.to(device)
        x7 = x7.to(device)
        x20 = m1.to(device).unsqueeze(axis = 1)
        x21 = m2.to(device).unsqueeze(axis = 1)
        x22 = m3.to(device).unsqueeze(axis = 1)
        x23 = m4.to(device).unsqueeze(axis = 1)
        x24 = m5.to(device).unsqueeze(axis = 1)
        x25 = m6.to(device).unsqueeze(axis = 1)
        x26 = m7.to(device).unsqueeze(axis = 1)
        x27 = m8.to(device).unsqueeze(axis = 1)
        
        combined = torch.cat((x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19),dim = -1)
        
        

        y_pred = model(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27)
        main = y_pred['label1']
        clock = y_pred['label2']
        
    
        loss = loss_fn(main,x4.float())
        loss1 = mse(clock,combined.float())
        
        loss = loss*0.8 + loss1*0.2
   
    
    
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

gt = []
pt = []
def evaluate2(model, loader, loss_fn,mse, device):
    gt.clear()
    pt.clear()
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for image,image1,image2,label,age,md,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,thickness,m1,m2,m3,m4,m5,m6,m7,m8 in loader:
            x1 = image.to(device)
            x2 = image1.to(device)
            x3 = image2.to(device)
            x4 = label.to(device).unsqueeze(axis =1)
            x5 = age.to(device).unsqueeze(axis =1)
            x6 = md.to(device).unsqueeze(axis =1)
            x7 = c1.to(device).unsqueeze(axis =1)
            x8 = c2.to(device).unsqueeze(axis =1)
            x9 = c3.to(device).unsqueeze(axis =1)
            x10 = c4.to(device).unsqueeze(axis =1)
            x11 = c5.to(device).unsqueeze(axis =1)
            x12 = c6.to(device).unsqueeze(axis =1)
            x13 = c7.to(device).unsqueeze(axis =1)
            x14 = c8.to(device).unsqueeze(axis =1)
            x15 = c9.to(device).unsqueeze(axis =1)
            x16 = c10.to(device).unsqueeze(axis =1)
            x17 = c11.to(device).unsqueeze(axis =1)
            x18 = c12.to(device).unsqueeze(axis =1)
            x19 = thickness.to(device).unsqueeze(axis =1)
            x20 = m1.to(device).unsqueeze(axis = 1)
            x21 = m2.to(device).unsqueeze(axis = 1)
            x22 = m3.to(device).unsqueeze(axis = 1)
            x23 = m4.to(device).unsqueeze(axis = 1)
            x24 = m5.to(device).unsqueeze(axis = 1)
            x25 = m6.to(device).unsqueeze(axis = 1)
            x26 = m7.to(device).unsqueeze(axis = 1)
            x27 = m8.to(device).unsqueeze(axis = 1)
            combined = torch.cat((x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19),dim = -1)

            y_pred = model(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27)
            main = y_pred['label1']
            clock = y_pred['label2']
          
            temp = y_pred['label1'].cpu()
            temp = torch.sigmoid(temp)
            temp = temp.reshape(-1).detach().numpy()
            temptrue = x4.cpu().detach().numpy()
            for i in temp:
                pt.append(i)
            for i in temptrue:
                gt.append(i)
        
        
            loss = loss_fn(main,x4.float())
            loss1 = mse(clock,combined.float())
            loss = loss*0.8+loss1*0.2
      
            epoch_loss += loss.item()
            
    
        epoch_loss = epoch_loss/len(loader)
    return epoch_loss
("")
def TrainModel2(numE,lr,seed):
    """ Seeding """
    seeding(seed)
    device = torch.device('cuda') 
    lowestloss = 1000
    model = AuxOut(True)
    model = model.to(device)
    counter = 0

  
    num_epochs = numE
    #lr = 1e-05
    lr = lr
    checkpoint_path = "C:/Users/Admin/Desktop/SeanWuGlaucomaProgression/baseline3odp/checkclockout" + str(seed) + ".pth"
  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, verbose=True,factor = 0.1)

 
    l = torch.as_tensor(3.18)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = l)
    
   # loss_fn = nn.BCEWithLogitsLoss()


    """ Training the model """
    best_auc = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train2(model, train_loader, optimizer, loss_fn,mse, device)
        valid_loss = evaluate2(model, val_loader, loss_fn,mse, device)

        """ Saving the model """
        score_auc = roc_auc_score(gt, pt)
        #precision, recall, thresholds = precision_recall_curve(gt, pt)
        #score_auc = auc(recall, precision)
        print(str(score_auc))
        
       
        if score_auc > best_auc:
            data_str = f"Valid ROCAUC loss improved from {best_auc:2.4f} to {score_auc:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_auc = score_auc
           # print("PRAuc: " + str(score_auc))
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
        if valid_loss >= lowestloss:
            print("valid loss increasing")
            counter +=1
        else:
            lowestloss = valid_loss
            print("Validation loss improved")
            counter = 0
        if counter > 10:
            print("early stopping")
           # break
       

mse = nn.MSELoss()

TrainModel2(100,0.0001,25)



# In[ ]:


print(lowestloss)


# In[51]:


import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

gt = []
pt = []
def evaluate2(model, loader, loss_fn,mse, device):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for image,image1,image2,label,age,md,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,thickness,m1,m2,m3,m4,m5,m6,m7,m8 in loader:
            x1 = image.to(device)
            x2 = image1.to(device)
            x3 = image2.to(device)
            x4 = label.to(device).unsqueeze(axis =1)
            x5 = age.to(device).unsqueeze(axis =1)
            x6 = md.to(device).unsqueeze(axis =1)
            x7 = c1.to(device).unsqueeze(axis =1)
            x8 = c2.to(device).unsqueeze(axis =1)
            x9 = c3.to(device).unsqueeze(axis =1)
            x10 = c4.to(device).unsqueeze(axis =1)
            x11 = c5.to(device).unsqueeze(axis =1)
            x12 = c6.to(device).unsqueeze(axis =1)
            x13 = c7.to(device).unsqueeze(axis =1)
            x14 = c8.to(device).unsqueeze(axis =1)
            x15 = c9.to(device).unsqueeze(axis =1)
            x16 = c10.to(device).unsqueeze(axis =1)
            x17 = c11.to(device).unsqueeze(axis =1)
            x18 = c12.to(device).unsqueeze(axis =1)
            x19 = thickness.to(device).unsqueeze(axis =1)
            x20 = m1.to(device).unsqueeze(axis = 1)
            x21 = m2.to(device).unsqueeze(axis = 1)
            x22 = m3.to(device).unsqueeze(axis = 1)
            x23 = m4.to(device).unsqueeze(axis = 1)
            x24 = m5.to(device).unsqueeze(axis = 1)
            x25 = m6.to(device).unsqueeze(axis = 1)
            x26 = m7.to(device).unsqueeze(axis = 1)
            x27 = m8.to(device).unsqueeze(axis = 1)

            y_pred = model(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27)
            main = y_pred['label1']
          
          
            temp = y_pred['label1'].cpu()
            temp = torch.sigmoid(temp)
            temp = temp.reshape(-1).detach().numpy()
            temptrue = x4.cpu().detach().numpy()
            for i in temp:
                pt.append(i)
            for i in temptrue:
                gt.append(i)
        
        
            loss = loss_fn(main,x4.float())
          
            epoch_loss += loss.item()
            
    
        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

checkpoint_path ="C:/Users/Admin/Desktop/SeanWuGlaucomaProgression/baseline3odp/clockout25.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AuxOut(True)
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

  
l = torch.as_tensor(3.18)
loss_fn = nn.BCEWithLogitsLoss(pos_weight = l)
mse = nn.MSELoss()


test_auc = evaluate2(model, test_loader, loss_fn,mse, device)
score_auc = roc_auc_score(gt, pt)
print(score_auc)
     


# In[60]:


wrong = []
count = 2
for i,j in zip(pt,gt):
    if(round(i)!=j):
        wrong.append(count)
    count +=1
print(wrong)


# In[63]:


pt = [round(x) for x in pt]
confMat = confusion_matrix(gt,pt)
    
disp = ConfusionMatrixDisplay(confusion_matrix=confMat)
disp.plot()


# In[59]:


pt = [round(x) for x in pt]
print(ConfusionMatrixDisplay(gt,pt))


# In[26]:


from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask


# In[42]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()
def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]




checkpoint_path = "C:/Users/Admin/Desktop/SeanWuGlaucomaProgression/baseline3odp/clockout25.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = AuxOut(True)
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()  
cam_extractor = SmoothGradCAMpp(model.model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



counter = 0
for image,image1,image2,label,age,md,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,thickness,m1,m2,m3,m4,m5,m6,m7,m8 in test_loader:
    x1 = image.to(device)
    x2 = image1.to(device)
    x3 = image2.to(device)
    x4 = label.to(device).unsqueeze(axis =1)
    x5 = age.to(device).unsqueeze(axis =1)
    x6 = md.to(device).unsqueeze(axis =1)
    x7 = c1.to(device).unsqueeze(axis =1)
    x8 = c2.to(device).unsqueeze(axis =1)
    x9 = c3.to(device).unsqueeze(axis =1)
    x10 = c4.to(device).unsqueeze(axis =1)
    x11 = c5.to(device).unsqueeze(axis =1)
    x12 = c6.to(device).unsqueeze(axis =1)
    x13 = c7.to(device).unsqueeze(axis =1)
    x14 = c8.to(device).unsqueeze(axis =1)
    x15 = c9.to(device).unsqueeze(axis =1)
    x16 = c10.to(device).unsqueeze(axis =1)
    x17 = c11.to(device).unsqueeze(axis =1)
    x18 = c12.to(device).unsqueeze(axis =1)
    x19 = thickness.to(device).unsqueeze(axis =1)
    optimizer.zero_grad()
    x6 = x6.to(device)
    x7 = x7.to(device)
    x20 = m1.to(device).unsqueeze(axis = 1)
    x21 = m2.to(device).unsqueeze(axis = 1)
    x22 = m3.to(device).unsqueeze(axis = 1)
    x23 = m4.to(device).unsqueeze(axis = 1)
    x24 = m5.to(device).unsqueeze(axis = 1)
    x25 = m6.to(device).unsqueeze(axis = 1)
    x26 = m7.to(device).unsqueeze(axis = 1)
    x27 = m8.to(device).unsqueeze(axis = 1)

    combined = torch.cat((x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19),dim = -1)



    y_pred = model(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27)
    out = y_pred['label1']

    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
          

    image2 = image2.reshape(-1, 224, 224)
    img = image2.cpu().detach().numpy().transpose(1, 2, 0)
    img = img[...,::-1].copy()
    
    plt.imshow(img)

    img = Image.fromarray((img*255).astype(np.uint8))
    

  #print(label.detach().cpu().numpy())
    new = torch.sigmoid(out)
    new = new.detach().cpu().numpy()
    new = new.round()
  
  
  

    if(label.detach().cpu().numpy() == 1 and new == 1):
        print("TRUE POSITIVE:")
        result = overlay_mask(img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.7)
        plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    
    elif (label.detach().cpu().numpy() == 0 and new == 0):
        print("TRUE NEGATIVE")
        result = overlay_mask(img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.7)
        plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    
    counter+=1


# In[43]:





# In[ ]:




