import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Conv2d, ReLU, Linear, Dropout2d

HIDDEN_NODES=512

class Manager(nn.Module):
    ''' Will HRL stabilize training so don't need double DQN etc? '''
    def __init__(self, in_chan, out_actions):
        super(Manager,self).__init__()
        self.conv1=Conv2d(in_chan,32,(8,8),stride=4)
        self.conv2=Conv2d(32,64,(4,4),2)
        self.conv3=Conv2d(64,32,(3,3),1)
        self.relu=ReLU()
        self.drop=Dropout2d(.5)
        self.fc1=Linear(36992,HIDDEN_NODES)
        self.fc2=Linear(HIDDEN_NODES,out_actions)
        self.loss=nn.CrossEntropyLoss()
        self.optim=optim.Adam(lr=1e-3)

    def forward(self,x):
        # import pdb; pdb.set_trace()
        x=self.drop(self.relu(self.conv1(x)))
        x=self.drop(self.relu(self.conv2(x)))
        x=self.drop(self.relu(self.conv3(x)))
        # torch.flatten(x,start_dim=1, end_dim=3)
        # print(x.shape)
        x=self.drop(self.relu(self.fc1(x.view(x.shape[0],-1))))
        x=self.fc2(x)
        return x

    def computeReward(self):
        pass

    def update(self,pred,truth):
        #I DON'T KNOW AND IT FREAKS ME OUT!!!
        pass



class Worker(nn.Module):
    def __init__(self, in_chan, out_actions):
        super(Worker,self).__init__()
        self.conv1 = Conv2d(in_chan, 32, (8, 8), stride=4)
        self.conv2 = Conv2d(32, 64, (4, 4), 2)
        self.conv3 = Conv2d(64, 32, (3, 3), 1)
        self.relu = ReLU()
        self.fc1 = Linear(36992, HIDDEN_NODES)
        self.fc2 = Linear(HIDDEN_NODES+out_actions, out_actions)

    def forward(self,x, goal):
        # import pdb; pdb.set_trace()
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.relu(self.conv3(x))
        # torch.flatten(x,start_dim=1)
        # print(x.shape)
        x=self.relu(self.fc1(x.view(x.shape[0],-1)))
        x=self.fc2(torch.cat((x,goal),1))
        return x

    def computeReward(self):
        pass

    def update(self):
        pass