import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

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
      

        t = F.relu(self.conv1(t)) #all <0 becomes 0
        t= F.max_pool2d(t, kernel_size = 2, stride = 2) #represents each region as its max i.e. 2x2 region --> 1x1 max val


        t = F.relu(self.conv2(t))
        t= F.max_pool2d(t, kernel_size = 2, stride = 2)

      

        t = F.relu(self.fc1(t.reshape(-1,192)))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t

def get_num_correct(predictions,targets):
    return predictions.argmax(dim=1).eq(targets).sum().item()


network = Network()

data_loader = torch.utils.data.DataLoader(train_set,batch_size=100) 
optimizer = optim.Adam(network.parameters(),lr=0.01)

total_loss = 0 
total_correct = 0
i = 0
losses = []
for epoch in range(5):

    for batch in data_loader:

        images,labels = batch #acquires training data and target from batch
        preds = network(images) #Transforms input into output throughout NN layers
        loss = F.cross_entropy(preds,labels) #Calculates cross entropy using predictions and labels 

        optimizer.zero_grad() #resets grad to 0 as pytorch does not reset them automatically
        loss.backward() #Calculates gradients through backward traversal of comp graph and adds them to previous gradient value
        optimizer.step() #Proceeds with gradient descent now that gradient is calculated

        total_loss += loss.item() #sums loss values throughout batches
        losses.append(loss.item())
        i = i+1
        total_correct += get_num_correct(preds,labels)#sums correct guesses throughout batches

    accuracy = total_correct/60000 * 100
    print("epoch   :",str(0), "total_correct: ",str(total_correct),"loss: ",str(total_loss))
    print("\n Accuracy = " + str(accuracy) + "%")

for j in range(1,len(losses)):
    print(str(losses[j])+ "\n")



