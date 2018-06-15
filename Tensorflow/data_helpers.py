import numpy as np
import re

import csv 
from sklearn.model_selection import StratifiedShuffleSplit                                                                                                                         
import numpy as np
#np.seed(0)


class data:
    def __init__(self):
        self.titles = []
        self.labels = []
        self.max_length = -1


    def read_data(self, file='../train.tsv'):
        i = 0
        with open(file, 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                #if i==0:
                #    continue
                if row[3] is not None and len(row[3]) > 0:
                    length = len(row[5].split())
                    if length > self.max_length:
                        self.max_length = length
                    self.titles.append(row[5])
                    self.labels.append([0, 1] if row[3][0] == 'N' else [1, 0])
                i += 1
        print('total number of labeled titles: ', len(self.titles))
        print('total number of labels: ', len(self.labels))
        return [np.array(self.titles), np.array(self.labels)], self.max_length

    def train_test_split(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

        for train_index, test_index in sss.split(self.titles, self.labels):
            self.titles_train, self.titles_test = np.array(self.titles)[train_index], np.array(self.titles)[test_index]
            self.labels_train, self.labels_test = np.array(self.labels)[train_index], np.array(self.labels)[test_index]
        print('train size:', len(self.titles_train) )
        print('test size:', len(self.titles_test))
        return self.titles_train, self.titles_test, self.labels_train, self.labels_test


    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()



    def batch_iter(self, data, batch_size, num_epochs, shuffle=False):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


