import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from joblib import load

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def search_trained_model():
  
  model = Net()

  model = torch.load('src/models/model.joblib')
  model.eval()

  return model

def get_model():
    model = search_trained_model()
    return model
  
async def predict(img):
  model = get_model()
  batch_size = 4
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  #Prepare image
  image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  image = cv2.resize(image,(32,32))
  image=transform(image)

  x_predict = torch.utils.data.DataLoader([image], batch_size=batch_size,shuffle=True, num_workers=2)
  dataiter = iter(x_predict)
  images= next(dataiter)
  images.shape
  output=model(images)
  print(output)
  _, predicted = torch.max(output, 1)
  predicted
  print(classes[predicted[0]])
  return classes[predicted[0]]
