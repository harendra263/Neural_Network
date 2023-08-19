import pandas as pd
import torch
from torch.utils.data import TensorDataset
from dataset import IrisDataset



def convert_target_to_numeric(df):
    labels = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    df["IrisType_num"] = df["Iris_Type"]
    df.IrisType_num = [labels[item] for item in df.IrisType_num]
    return df


def convert_to_tensor(df):  # sourcery skip: avoid-builtin-shadow
    input = df.iloc[:, 1:-2]
    print("\nInput values are:")
    print(input.head())
    output = df.loc[:, 'IrisType_num']
    print("\nThe output value is:")
    print(output.head())
    input = torch.Tensor(input.to_numpy())
    print("\nInput format: ", input.shape, input.dtype)
    output = torch.Tensor(output.to_numpy())
    print("Output format: ", output.shape, output.dtype)
    return TensorDataset(input, output), input, output


def get_tensors() -> TensorDataset:
    data = IrisDataset("iris/Iris_dataset.xlsx", file_type="xlsx", target="Iris_Type").get_dataframe()
    df = convert_target_to_numeric(df=data)
    tensor_dataset, input, _ = convert_to_tensor(df=df)
    return tensor_dataset, input

