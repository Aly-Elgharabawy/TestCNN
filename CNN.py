import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
)

class Network(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 192, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)

    def forward(self,t):
        t = F.relu(self.conv1(t))
        t= F.max_pool2d(t, kernel_size = 2, stride = 2)

        t = F.relu(self.conv2(t))
        t= F.max_pool2d(t, kernel_size = 2, stride = 2)

        t = F.relu(self.fc1(t.reshape(-1,192)))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t

network = Network()

sample = next(iter(train_set)) 
image, label = sample
pred = network(image.unsqueeze(0)) # unsqueeze 0 ---> simulate batch size i.e. 3d to 4d
print(pred)




