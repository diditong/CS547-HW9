'''
CS 547 Homework 3
Name: Jiashuo Tong
Date: 10/02/2019
'''

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as functions
import torch.optim as optim
from torch.autograd import Variable

'''
PART1: PREPARE DATA USING THE DATA AUGMENTATION TECHNIQUE
'''

# Define transformations for Data Augmentation
trans_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.1), # Horizontally flip images
                                  transforms.RandomVerticalFlip(p=0.1), # Vertically flip images                       
                                  transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Normalize data
trans_test = transforms.Compose([transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Normalize data

# Download CIFAR10 data to directory ./CIFAR10
train = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=trans_train)
test = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=trans_test)

# Prepare data loader for training purposes
data_train = torch.utils.data.DataLoader(train, batch_size=100)
data_test = torch.utils.data.DataLoader(test, batch_size=100)

'''
PART2: DEFINE THE LAYERS AND BUILD THE CNN
'''

# Define the CNNModel as a class
class CNNModel(nn.Module):
    
    # Define the layers as building blocks for the CNN
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Define the convolution layers
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0)     
        self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0) 
        self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0)  
        
        # Define the fully connected layers
        self.fc_layer1 = nn.Linear(in_features=1024, out_features= 500) 
        self.fc_layer2 = nn.Linear(in_features=500, out_features= 500)
        self.fc_layer3 = nn.Linear(in_features=500, out_features= 10)      
        
        # Define the batch normalization layers
        self.bn_layer1 = nn.BatchNorm2d(64)
        self.bn_layer2 = nn.BatchNorm2d(64)
        self.bn_layer3 = nn.BatchNorm2d(64)
        self.bn_layer4 = nn.BatchNorm2d(64)
        self.bn_layer5 = nn.BatchNorm2d(64)
        
        # Define the dropout layers 
        self.dropout = nn.Dropout2d(p=0.5)
        
        # Define the max pooling layers
        self.pool = nn.MaxPool2d(2, stride=2)
        
    # Build the CNN structure proposed in Lecture 6 Note
    def forward(self, X):
        X = functions.relu(self.conv_layer1(X)) # Relu as the activation function
        X = self.bn_layer1(X) # Batch normalization
        X = functions.relu(self.conv_layer2(X))
        X = self.pool(X) # Pooling layer
        X = self.dropout(X) # Perform dropout
        X = functions.relu(self.conv_layer3(X))
        X = self.bn_layer2(X)
        X = functions.relu(self.conv_layer4(X))
        X = self.pool(X)
        X = self.dropout(X)
        X = functions.relu(self.conv_layer5(X))
        X = self.bn_layer3(X)
        X = functions.relu(self.conv_layer6(X))
        X = self.dropout(X)
        X = functions.relu(self.conv_layer7(X))
        X = self.bn_layer4(X)
        X = functions.relu(self.conv_layer8(X)) 
        X = self.bn_layer5(X)
        X = self.dropout(X)
        X = X.view(-1, 32*32) # Reshape the current layer using view() function
        X = functions.relu(self.fc_layer1(X))
        X = functions.relu(self.fc_layer2(X))
        X = self.fc_layer3(X)
        return X

'''
PART3: PREPARE FOR TRAINING
'''

model = CNNModel() # Define a model using the constructor CNNModel()
print ('The model has the following structure:\n', model)
model.cuda() # Send model to GPU
batch_size = 100 # Define batch size
num_epochs = 100 # Define the total number of epochs
learning_rate = 1e-3 # Set the learning rate to be 1e-3
F_loss = nn.CrossEntropyLoss() # Define the cross entropy loss function (negative log)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Use the ADAM Optimizer on the trainable parameters
model.train() # Model should be set to the training mode
training_accuracy = []

'''
PART4: TRAIN THE MODEL
'''

for epoch in range(num_epochs):
    for i, data in enumerate(data_train, 0):
        images, labels = data # Separate images and labels from data
        images, labels = Variable(images).cuda(), Variable(labels).cuda() # Send data to GPU
        optimizer.zero_grad() # Zero out the gradients
        output = model(images) # Feed forward
        loss = F_loss(output, labels) # Compute the loss
        loss.backward() # Back Propagate
        optimizer.step() # Update optimizer
        prediction = output.max(1)[1] # Extract the indices information (predictions) from the output
        accuracy = (prediction.eq(labels).sum())/batch_size*100.0 # Compute the training accuracy for the current batch
        training_accuracy.append(accuracy) # Append the batch accuracy to the training accuracy list
    epoch_accuracy = np.mean(training_accuracy) # Compute accuracy for the epoch
    print('\nEpoch ', epoch,'/',num_epochs,' is complete with an accuracy of', epoch_accuracy)

'''
EXTRA CREDIT: COMPARE HERISTIC WITH MONTE CARLO
'''

# First, we implement the heuristic method
num_iterations_heuristic = 1 # One iteration for Heuristic
test_accuracy = []
model.eval()
    
for data in data_test:
    images, labels = data
    images, labels = Variable(images).cuda(), Variable(labels).cuda() # Send data to GPU
    output = model(images) # Feed forward
    prediction = output.max(1)[1] # Extract the indices information (predictions) from the output
    accuracy = (prediction.eq(labels).sum())/batch_size*100.0 # Compute the test accuracy
    test_accuracy.append(accuracy) # Append the batch accuracy to the test accuracy list
test_accuracy = np.mean(test_accuracy)
print('With Heuristic method, accuracy on the test set is ', test_accuracy)

# Next, we implement the MonteCarlo method
num_iterations_MonteCarlo = 100 # Set one hundred iterations for MonteCarlo
test_accuracy = []

for data in data_test:
    images, labels = data
    images, labels = Variable(images), Variable(labels)
    for it in range (num_iterations_MonteCarlo):
        if it == 0:
            output = model(images)
        else:
            output += model(images)
    output /= num_iterations
    prediction = output.max(1)[1] 
    accuracy = (prediction.eq(labels).sum())/batch_size*100.0 # Compute the test accuracy
    test_accuracy.append(accuracy) # Append the batch accuracy to the test accuracy list
test_accuracy = np.mean(test_accuracy)
print('With Monte Carlo simulation, accuracy on the test set is ', test_accuracy)