import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import collections

from dataset import train_dataset, test_dataset

print(len(train_dataset))

# print(train_dataset[0])

print(type(train_dataset[0]))

# input Matrix

print(train_dataset[0][0].size())  
# Output:- torch.Size([1, 28, 28]) means It simply defines an image of size 28x28 has 1 channel, 
# which means it's a grayscale image. If it was a colored image then instead of 1 there would be 
# 3 as the colored image has 3 channels such as RGB.
# In case of torch.Size([64, 1, 28, 28]) 64 is the no of images

print(train_dataset[0][1])

show_img = train_dataset[2][0].numpy().reshape(28,  28)
# plt.imshow(show_img, cmap='gray')
# plt.show()

BATCH_SIZE = 100
N_ITERS = 3000

num_epochs = N_ITERS / (len(train_dataset) / BATCH_SIZE)
num_epochs = int(num_epochs)
print(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = BATCH_SIZE, shuffle=False)

print(isinstance(train_loader, collections.abc.Iterable))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim) ->None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    

# sourcery skip: avoid-builtin-shadow
input_dim = 28*28
output_dim=10

model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)

# Creating Cross Entropy Loss Class
criterion = nn.CrossEntropyLoss()

# Create Optimizer

LEARNING_RATE = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

iter = 0
for _ in range(num_epochs):
    for _, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28).requires_grad_()
        labels = labels

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        #clear the gradient
        optimizer.zero_grad()
        iter +=1

        if iter % 450 == 0:
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = images.view(-1, 28 * 28).requires_grad_()
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                # total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct.item() / total

            # Print loss
            print(f"Iteration: {iter}, Loss: {loss.item()}, Accuracy: {accuracy}")









