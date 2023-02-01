import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNQuantized(nn.Module):
    def __init__(self):
        super().__init__()
        #1st conv block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.dropout_conv = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)

        #2nd conv block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        #3rd conv block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)

        #4th conv block
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(32)

        #5th conv block
        self.conv5 = nn.Conv2d(256, 128, 3, padding=1)

        #6th conv block
        self.conv6 = nn.Conv2d(128, 64, 3, padding=1)

        #7th conv block
        self.conv7 = nn.Conv2d(64, 32, 3, padding=1)

        #dropout and fully connected layer
        self.dropout_fc = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4960, 21)
        

    #forward propogation
    def forward(self, x):
    	#1st conv block
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #2nd conv block
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)

        #3rd conv block
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #4th conv block
        x = F.relu(self.conv4(x))
        x = self.dropout_conv(x)

        #5th conv block
        x = F.relu(self.conv5(x))
        x = self.dropout_conv(x)

        #6th conv block
        x = F.relu(self.conv6(x))
        x = self.dropout_conv(x)

        #7th conv block
        x = F.relu(self.conv7(x))
        x = self.dropout_conv(x)

        #dropout and fully connected layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout_fc(x)
        x = self.fc1(x)
        return x



class ConvNet_580k(nn.Module):
    def __init__(self):
        super().__init__()
        #1st conv block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.dropout_conv = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)

        #2nd conv block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        #3rd conv block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)

        #4th conv block
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(32)

        #5th conv block
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)

        #6th conv block
        self.conv6 = nn.Conv2d(128, 64, 3, padding=1)

        #7th conv block
        self.conv7 = nn.Conv2d(64, 32, 3, padding=1)

        #dropout and fully connected layer
        self.dropout_fc = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4960, 21)
        

    #forward propogation
    def forward(self, x):
        #1st conv block
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #2nd conv block
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)

        #3rd conv block
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #4th conv block
        x = F.relu(self.conv4(x))
        x = self.dropout_conv(x)

        #5th conv block
        x = F.relu(self.conv5(x))
        x = self.dropout_conv(x)

        #6th conv block
        x = F.relu(self.conv6(x))
        x = self.dropout_conv(x)

        #7th conv block
        x = F.relu(self.conv7(x))
        x = self.dropout_conv(x)

        #dropout and fully connected layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout_fc(x)
        x = self.fc1(x)
        return x


class ConvNet_300k(nn.Module):
    def __init__(self):
        super().__init__()
        #1st conv block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.dropout_conv = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)

        #2nd conv block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        #3rd conv block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)

        #4th conv block
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(32)

        #5th conv block
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)

        #6th conv block
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)

        #7th conv block
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)

        #dropout and fully connected layer
        self.dropout_fc = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4960, 21)
        

    #forward propogation
    def forward(self, x):
        #1st conv block
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #2nd conv block
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)

        #3rd conv block
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #4th conv block
        x = F.relu(self.conv4(x))
        x = self.dropout_conv(x)

        #5th conv block
        x = F.relu(self.conv5(x))
        x = self.dropout_conv(x)

        #6th conv block
        x = F.relu(self.conv6(x))
        x = self.dropout_conv(x)

        #7th conv block
        x = F.relu(self.conv7(x))
        x = self.dropout_conv(x)

        #dropout and fully connected layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout_fc(x)
        x = self.fc1(x)
        return x


class ConvNet_144k(nn.Module):
    def __init__(self):
        super().__init__()
        #1st conv block
        self.conv1 = nn.Conv2d(1, 24, 3, padding=1)
        self.dropout_conv = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)

        #2nd conv block
        self.conv2 = nn.Conv2d(24, 24, 3, padding=1)

        #3rd conv block
        self.conv3 = nn.Conv2d(24, 32, 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)

        # #4th conv block
        # self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        # # self.bn4 = nn.BatchNorm2d(32)

        #5th conv block
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)

        #6th conv block
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)

        #7th conv block
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)

        #dropout and fully connected layer
        self.dropout_fc = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4960, 21)
        

    #forward propogation
    def forward(self, x):
        #1st conv block
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #2nd conv block
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)

        #3rd conv block
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        # #4th conv block
        # x = F.relu(self.conv4(x))
        # x = self.dropout_conv(x)

        #5th conv block
        x = F.relu(self.conv5(x))
        x = self.dropout_conv(x)

        #6th conv block
        x = F.relu(self.conv6(x))
        x = self.dropout_conv(x)

        #7th conv block
        x = F.relu(self.conv7(x))
        x = self.dropout_conv(x)

        #dropout and fully connected layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout_fc(x)
        x = self.fc1(x)
        return x




class ConvNet_200k(nn.Module):
    def __init__(self):
        super().__init__()
        #1st conv block
        self.conv1 = nn.Conv2d(1, 26, 3, padding=1)
        self.dropout_conv = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)

        #2nd conv block
        self.conv2 = nn.Conv2d(26, 26, 3, padding=1)

        #3rd conv block
        self.conv3 = nn.Conv2d(26, 32, 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)

        #4th conv block
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(32)

        #5th conv block
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)

        #6th conv block
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)

        #7th conv block
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)

        #dropout and fully connected layer
        self.dropout_fc = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4960, 21)
        

    #forward propogation
    def forward(self, x):
        #1st conv block
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #2nd conv block
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)

        #3rd conv block
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #4th conv block
        x = F.relu(self.conv4(x))
        x = self.dropout_conv(x)

        #5th conv block
        x = F.relu(self.conv5(x))
        x = self.dropout_conv(x)

        #6th conv block
        x = F.relu(self.conv6(x))
        x = self.dropout_conv(x)

        #7th conv block
        x = F.relu(self.conv7(x))
        x = self.dropout_conv(x)

        #dropout and fully connected layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout_fc(x)
        x = self.fc1(x)
        return x