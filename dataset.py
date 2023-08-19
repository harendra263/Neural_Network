import pandas as pd

class IrisDataset:
    def __init__(self, data_path: str, file_type: str, target: str) -> None:
        if file_type == "csv":
            self.data = pd.read_csv(data_path)
        else:
            self.data = pd.read_excel(data_path)
        
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            row = self.data.iloc[index]
            if self.target not in row:
                raise ValueError("Target column not found in row")
            target_value = row[self.target]
            data = row.drop(self.target)
            return data, target_value
        elif isinstance(index, str):
            return self.data[index]
        else:
            raise TypeError("Index must be integer or a string column name")
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.data
