from torch.utils.data import DataLoader, random_split
from exploratory import get_tensors
import config


tensor_dataset, input = get_tensors()

def split_df(tensor_df = tensor_dataset, input = input):
    number_rows = len(input)
    test_split = int(number_rows*0.3)
    validate_split = int(number_rows*0.2)
    train_split = number_rows - test_split - validate_split
    train_set, validate_set, test_set = random_split(
        tensor_df, [train_split, validate_split, test_split]
    )
    train_loader = DataLoader(train_set, batch_size= config.TRAIN_BATCH_SIZE, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=config.VALID_BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=config.TEST_BATCH_SIZE)
    
    return train_loader, validate_loader, test_loader
 