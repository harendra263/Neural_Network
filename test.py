import torch
from model import Network
import config
import dataloader

_, _, test_loader = dataloader.split_df()

labels = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}


def test():
    # Load the model that we saved at the end of the training loop
    model = Network(config.INPUT_SIZE, config.OUTPUT_SIZE)
    path = "model/model.pth"
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
        
        print("Accuracy of the model %d %%" %(100* running_accuracy / total))


def test_species():
    model = Network(input_size=config.INPUT_SIZE, output_size=config.OUTPUT_SIZE)
    path = "model/model.pth"
    model.load_state_dict(torch.load(path))

    labels_length = config.OUTPUT_SIZE   # 3
    labels_correct = [0 for _ in range(labels_length)]
    labels_total = [0 for _ in range(labels_length)]

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            print("outputs:", outputs)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)

            label_correct_running = (predicted == outputs).squeeze()
            label =  int(outputs[0].item())
            if label_correct_running.item():
                labels_correct[label] += 1
            labels_total[label] += 1

    label_list = list(labels.keys())
    for i in range(config.OUTPUT_SIZE):
        print("Accuracy to predict %5s : %2d %%" % (label_list[i], 100 * labels_correct[i] / labels_total[i]))


if __name__ == "__main__":
    test()
    test_species()