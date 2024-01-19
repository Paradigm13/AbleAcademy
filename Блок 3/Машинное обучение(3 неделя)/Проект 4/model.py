import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = None # словарь, используемый для преобразования меток в числа
        self.num2label = None # словарь, используемый для преобразования числа в метки
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", 
        чтобы заполнить все атрибуты данного класса.
        '''
        # Начало вашего кода
        self._train_X, self._train_y = dataset.train
        self._val_X, self._val_y = dataset.val
        self._test_X, self._test_y = dataset.test
        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        
        for x in self._train_X:
            self.vocab.update(set(x.split()))
            
        self.spam = {word: 0 for word in self.vocab}    
        self.ham = {word: 0 for word in self.vocab}
        
        for i in range(len(self._train_X)):
            x = self._train_X[i]    
            y = self._train_y[i]
            words = x.split()
            
            if y == self.num2label[1]:
                for word in words:
                    self.spam[word] += 1
        #  self.spam.get(word, 0) + 1           
            else:
                for word in words:
                    self.ham[word] += 1
        #  self.ham.get(word, 0) + 1                
        self.Nvoc = len(self.vocab)
        self.Nspam = sum(self.spam.values())
        self.Nham = sum(self.ham.values())    
        
        # Конец вашего кода
        pass
    
    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''
        # Начало вашего кода
        
        p_spam = np.sum(self._train_y == 'spam') / len(self._train_y)
        p_ham = 1 - p_spam
        
        # message = np.char.lower(message.astype(str))
        # message = np.array([re.sub('[^\w\s]', ' ', x) for x in message.astype(str)])
        message = message.astype(str).split()
        deliminator_spam = (self.Nspam + self.alpha * self.Nvoc)
        deliminator_ham = (self.Nham + self.alpha * self.Nvoc)
        pspam = 1
        pham = 1
        for word in message:
            if word in self.spam and self.spam[word] > 0:
                pspam *= (self.spam[word] + self.alpha) / deliminator_spam
            else:
                pspam *= (0 + self.alpha) / deliminator_spam
                
            if word in self.ham and self.ham[word] > 0:
                pham *= (self.ham[word] + self.alpha) / deliminator_ham
            else:
                pham *= (0 + self.alpha) / deliminator_ham
                
        pspam = pspam * p_spam
        pham = pham * p_ham      
        # Конец вашего кода
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        correct = 0
        total_val = len(self._val_X)
        
        for i in range(total_val):
            x = self._val_X[i]
            y = self._val_y[i]
            pred = self.inference(x)
            
            if pred == y:
                correct += 1
        val_acc = correct / total_val
        # Конец вашего кода
        return val_acc 

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        correct = 0
        total_test = len(self._test_X)
        
        for i in range(total_test):
            x = self._test_X[i]
            y = self._test_y[i]
            pred = self.inference(x)
            
            if pred == y:
                correct += 1
        test_acc = correct / total_test
        # Конец вашего кода
        return test_acc


