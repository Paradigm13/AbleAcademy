import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {} # словарь, используемый для преобразования меток в числа
        self.num2label = {} # словарь, используемый для преобразования числа в метки
        self._transform()
        
    def __len__(self):
        
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Начало вашего кода
        self._x = np.array([re.sub('[^\w\s]', ' ', x) for x in self._x.astype(str)])
        self._x = np.array([' '.join(s.split()) for s in self._x.astype(str)])
        
        labels = np.unique(self._y)
        num_label = np.arange(len(labels))
        self.label2num = dict(zip(labels, num_label))
        self.num2label = dict(zip(num_label, labels))
        # self._y = np.array([self.label2num[label] for label in self._y])    
     
        # Конец вашего кода
        pass

    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        # Начало вашего кода
        indices = np.random.permutation(len(self._x))
        val_indices = indices[:int(len(self._x) * val)]
        test_indices = indices[int(len(self._x) * val):int(len(self._x) * (val + test))]
        train_indices = indices[int(len(self._x) * (val + test)):]
        
        self.train = (self._x[train_indices], self._y[train_indices])
        self.val = (self._x[val_indices], self._y[val_indices])
        self.test = (self._x[test_indices], self._y[test_indices])
        # Конец вашего кода
        pass
