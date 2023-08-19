# sourcery skip: avoid-builtin-shadow
import dataset

df = dataset.IrisDataset("iris/Iris_dataset.xlsx", file_type="xlsx", target="Iris_Type").get_dataframe()

INPUT_SIZE = df.drop(['#', "Iris_Type"], axis=1).shape[1]
OUTPUT_SIZE = df["Iris_Type"].nunique()
DEVICE = "cuda"
TRAIN_BATCH_SIZE = 10
VALID_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
LEARNING_RATE = 0.01
EPOCHS= 50
MODEL_PATH= "model/model.pth"
