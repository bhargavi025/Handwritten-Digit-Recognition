import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device",device)

transform = transforms.ToTensor()
print("Using device", device)

#load data
train_data = MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform   
    )

test_data =MNIST(
    root = "data",
    train = False,
    download = True,
    transform=transform
    ) 

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True   
    )

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
    )

#create fully connected network
class DigitM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(28*28,128), #hidden layer
            nn.ReLU(), #Activation function
            
            nn.Linear(128,64), #hidden layer 
            nn.ReLU(), #activation function
            
            nn.Linear(64,10)  #output layer          
            )
    
    def forward(self, x):
        return self.net(x)

#initialize network
model = DigitM().to(device)

#loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

#train the network
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        #get correct shape of data 
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        
        #forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        #backward
        loss.backward()
        optimizer.zero_grad()
        #gradient descent
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epoch} | Loss: {total_loss:.4f}")
    
#check accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)


accuracy = 100 * correct/total
print(f"Total Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "mnist.pth")
print("Model Saved as mnist.pth")

index = 0
image, true_label = test_data[index]

plt.imshow(image.squeeze(), cmap = "gray")
plt.title(f"Actual Label : {true_label}")
plt.axis("off")
plt.show()

image_flat = image.view(1, -1).to(device)

with torch.no_grad():
    output = model(image_flat)
    predicted_label = output.argmax(dim=1).item()

print ("User Picked Image Index", index) 
print("Actual Label", true_label) 
print("Model prediction", predicted_label)
        


























