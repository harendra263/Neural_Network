# sourcery skip: avoid-builtin-shadow
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn import datasets
import pandas as pd

# Loading the dataset
df = pd.read_csv("iris/Iris.csv")
print(df.head())
print(df.Species.value_counts())

# Convert Iris species into numeric types: 
labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['IrisType_num'] = df.Species
df['IrisType_num'] = df['IrisType_num'].map(labels)
print(df.head())

input = df.iloc[:, 1:-2]
print(input.head())

output = df.loc[:, "IrisType_num"]
print(output.head())


# To train the model we need to convert the input and output to tensor

input = torch.tensor(input.to_numpy())
input = input.type(torch.float32)
print('\nInput format: ', input.shape, input.dtype) 

ouput = torch.tensor(output.to_numpy())
print('Output format: ', output.shape, output.dtype)


data = TensorDataset(input, ouput)

# Split to Train, validate and test using random_split

train_batch_size = 10
number_rows = len(input)
test_split = int(number_rows * 0.3)
validate_split = int(number_rows * 0.2)

train_split = number_rows - test_split - validate_split

train_set, validate_set, test_set = random_split(data, [train_split, validate_split, test_split])

# Create DataLoader to read the data within and put into memory
train_loader = DataLoader(train_set, batch_size= train_batch_size, shuffle=True)
validate_loader = DataLoader(validate_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)

# Struture of the model:

# Linear -> ReLU -> Linear -> ReLU -> Linear

input_size = list(input.shape)[1] # # = 4. The input depends on how many features we initially feed the model. In our case, there are 4 features for every predict value
learning_rate = 0.01
output_size = len(labels) # The output is prediction results for three types of Irises.

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(input_size, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, output_size)

    
    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        return self.layer3(x2)
    

# Instantiate the model
model = NeuralNetwork(input_size=input_size, output_size=output_size)

# Define your execution device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device\n")
model.to(device=device)

# Function to save model
def savemodel():
    path = "models/NetModel.pth"
    torch.save(model.state_dict(), path)

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Training
def train(num_epochs: int):
    best_accuracy = 0.0

    print("Begin Training...")
    for epoch in range(1, num_epochs+1):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0

        # Training Loop
        for data in train_loader:
            inputs, outputs = data # get the input and real species as outputs; data is a list of [inputs, outputs] 
            optimizer.zero_grad()
            predicted_outputs = model(inputs)
            train_loss = loss_fn(predicted_outputs, outputs)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        
        # Calculate the training loss value
        train_loss_value = running_train_loss / len(train_loader)

        # Validation loop
        with torch.no_grad():
            model.eval()
            for data in validate_loader:
                inputs, outputs = data
                predicted_outputs = model(inputs)
                val_loss = loss_fn(predicted_outputs, outputs)

                # The label with the highest value will be our prediction 
                _, predicted = torch.max(predicted_outputs, 1)
                running_val_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

        # Caculate validation loss value
        val_loss_value = running_val_loss / len(validate_loader)

        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.
        accuracy = (100 * running_accuracy / total)

        # save the model if accuracy is best
        if accuracy > best_accuracy:
            savemodel()
            best_accuracy = accuracy
        
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, 'Accuracy is %d %%' % (accuracy))


def test():
    # Load the model we save at the end of training loop
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    path = "models/NetModel.pth"
    model.load_state_dict(torch.load(path))

    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()
        print('Accuracy of the model based on the test set of', test_split ,'inputs is: %d %%' % (100 * running_accuracy / total))    

# Optional: Function to test which species were easier to predict

# Optional: Function to test which species were easier to predict  
def test_species(): 
    # Load the model that we saved at the end of the training loop 
    model = NeuralNetwork(input_size, output_size)
    path = "models/NetModel.pth"
    model.load_state_dict(torch.load(path)) 

    labels_length = len(labels) # how many labels of Irises we have. = 3 in our database. 
    labels_correct = [0. for _ in range(labels_length)]
    labels_total = [0. for _ in range(labels_length)]

    with torch.no_grad(): 
        for data in test_loader: 
            inputs, outputs = data 
            predicted_outputs = model(inputs) 
            _, predicted = torch.max(predicted_outputs, 1) 

            label_correct_running = (predicted == outputs).squeeze() 
            label = outputs[0] 
            if label_correct_running.item():  
                labels_correct[label] += 1 
            labels_total[label] += 1  

    label_list = list(labels.keys())
    for i in range(output_size): 
        print('Accuracy to predict %5s : %2d %%' % (label_list[i], 100 * labels_correct[i] / labels_total[i]))




if __name__ == "__main__":
    num_epochs = 25
    train(num_epochs=num_epochs)
    print("Finished Training\n")
    test()
    test_species()









