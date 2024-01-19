import pandas as pd
import numpy as np
from dataset import Dataset
# from model import Model

data = pd.read_csv('./spam.csv', encoding = "Windows-1252")
data = data.iloc[:,:2]
data.head()


X = np.array(data['v2'])
y = np.array(data['v1'])

dataset = Dataset(X,y)

from model import Model

dataset.split_dataset(val=0.1, test=0.1)
model1 = Model()
model1.fit(dataset)

validation_prob = model1.validation()
print(f"Точность классификации сообщений для validation: {validation_prob*100:.2f} %")
# dataset.split_dataset(val=0.1, test=0.1)
# model1 = Model()
# model1.fit(dataset)