import torch
from torch import optim
from torchvision import datasets, transforms, utils
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Modified example work script from implimenting Convolutional Neural Networks as a feature extractor in PyTorch.
# This script is intended as a work example of Adam Morphy's work with the Vancouver Whitecaps FC, and has been modified outside its original data pipeline.

#################################################
#                   _______
#                  |       |              
#            o     |       |
#          -()-   o
#           |\
#################################################

# Custom layer
class PixelLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.x = x
    
    def forward(self, x):
        surface = x[:,0,:,:]
        mask = x[:,1,:,:]
        masked = surface * mask
        value = torch.sum(masked, dim=(2,1))
        return value

    
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 16, 3, padding='same')
        self.maxpool = nn.MaxPool2d(2) # 
        self.conv2 = nn.Conv2d(16, 1, 1) # repeats
        self.conv3 = nn.Conv2d(1, 32, 3, padding='same') # verify input channels
        self.conv4 = nn.Conv2d(32, 1, 1) # repeats
        self.conv5 = nn.Conv2d(1, 16, 3, padding='same') # verify input channels
        self.conv6 = nn.Conv2d(16, 1, 1)
        torch.nn.init.constant_(self.conv6.weight, avg_completion_rate)
        self.upsample = nn.UpsamplingNearest2d(size=(104, 68))
        self.pixelval = PixelLayer()

    def forward(self, x, inp_data):
        x = F.relu(self.conv1(x))
        x = self.conv2(x) 
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x) 
        x = self.upsample(x)
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        combined = torch.concat((x, inp_data), dim=1)
        pixel = self.pixelval(combined)

        return combined, pixel
    
def trainer(model, criterion, optimizer, trainloader, validloader, epochs=5, verbose=True):
    """Simple training wrapper for PyTorch network."""
    
    train_loss, valid_loss, valid_accuracy = [], [], []
    for epoch in range(epochs):  # for each epoch
        train_batch_loss = 0
        valid_batch_loss = 0
        valid_batch_acc = 0
        
        # Training
        for Xp, Xd, y in trainloader:
            optimizer.zero_grad()       # Zero all the gradients w.r.t. parameters
            _, y_hat = model(Xp, Xd)           # Forward pass to get output
            loss = criterion(y_hat, y)  # Calculate loss based on output
            loss.backward()             # Calculate gradients w.r.t. parameters
            optimizer.step()            # Update parameters
            train_batch_loss += loss.item()  # Add loss for this batch to running total
        train_loss.append(train_batch_loss / len(trainloader))
        
        # Validation
        model.eval()
        with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
            for Xp, Xd, y in validloader:
                _, y_hat = model(Xp, Xd)
                y_hat_labels = y_hat > 0.5 #torch.softmax(y_hat, dim=1).topk(1, dim=1)
                loss = criterion(y_hat, y)
                valid_batch_loss += loss.item()
                valid_batch_acc += (y_hat_labels.squeeze() == y).type(torch.float32).mean().item()
        valid_loss.append(valid_batch_loss / len(validloader))
        valid_accuracy.append(valid_batch_acc / len(validloader))  # accuracy
        
        model.train()
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch + 1}:",
                  f"Train Loss: {train_loss[-1]:.3f}.",
                  f"Valid Loss: {valid_loss[-1]:.3f}.",
                  f"Valid Accuracy: {valid_accuracy[-1]:.2f}.")
    
    results = {"train_loss": train_loss,
               "valid_loss": valid_loss,
               "valid_accuracy": valid_accuracy}
    return results, model


# FIT MODEL
model = ConvNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
results, trained_model = trainer(model, criterion, optimizer, trainloader, validloader, epochs=10)

# MAKE PREDICTIONS
trained_model.eval()
with torch.no_grad():
    surfaces = trained_model(Xp_test, Xd_test)
    
surface_prob, _ = surfaces