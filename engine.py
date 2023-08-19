import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from model import Network
import config
import dataloader

train_loader, validate_loader, _ = dataloader.split_df()

model = Network(input_size=config.INPUT_SIZE, output_size=config.OUTPUT_SIZE)

def save_model():
    path = config.MODEL_PATH
    torch.save(model.state_dict(), path)

loss_fn =  nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.0001)


def train(num_epochs):  # sourcery skip: hoist-statement-from-loop
    best_accuracy = 0.0

    print("Begin training....")
    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total =0

        # training loop
        for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, outputs = data
            optimizer.zero_grad()
            predicted_outputs = model(inputs)
            train_loss = loss_fn(predicted_outputs, outputs.long())
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        
        #calculate training loss value
        train_loss_value=  running_train_loss/len(train_loader)

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for idx, data in tqdm(enumerate(validate_loader), total=len(validate_loader)):
                inputs, outputs = data
                predicted_outputs = model(inputs)
                val_loss = loss_fn(predicted_outputs, outputs.long())

                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs, 1)
                running_val_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()
        
        val_loss_value = running_val_loss/ len(validate_loader)

        accuracy = (100 * running_accuracy / total)

        if accuracy > best_accuracy:
            save_model()
            best_accuracy = accuracy

        print("Completed training batch", epoch, "Training Loss is: %4f" %train_loss_value, "Validation Loss is: %.4f" %val_loss_value, "Accuracy is %d %%" %(accuracy))


if __name__ == "__main__":
    num_epochs = config.EPOCHS
    train(num_epochs=num_epochs)
    print("finished training \n")
