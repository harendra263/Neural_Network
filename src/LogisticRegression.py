import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import train_dataset

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
plt.imshow(show_img, cmap='gray')
plt.show()







