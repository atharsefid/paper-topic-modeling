import csv
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
#np.seed(0)


class data:
    def __init__(self):
        self.titles = []
        self.labels = []
        self.max_length = -1
        self.test_titles = []
        self.test_lables = []
        self.abstracts = []   
        self.max_abs_length = -1
    def read_data(self, file, title_index= 5, label_index= 3):
        i = 0
        with open(file, 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                if row[3] is not None and len(row[3]) > 0:
                    length = len(row[5].split())
                    if length > self.max_length:
                        self.max_length = length
                    self.titles.append(row[5])
                    self.labels.append(0 if row[3][0] == 'N' else 1)
                i += 1
        print('total number of labeled titles: ', len(self.titles))
        print('total number of labels: ', len(self.titles))
        return self.max_length

    def read_test(self, file, title_index= 3, label_index= 1):
        i = 0
        with open(file, 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                if row[1] is not None and len(row[1]) > 0:
                    self.test_titles.append(row[3])
                    self.test_lables.append(0 if row[1][0] == 'N' else 1)
                i += 1
        print('total number of labeled titles: ', len(self.titles))
        print('total number of labels: ', len(self.titles))
        return self.test_titles, self.test_lables 
    
    def read_abstracts(self, file):
        abst = {}
        with open('papers.tsv', 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                abst[row[0]] = row[8]
        i = 0
        with open(file, 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                if row[3] is not None and len(row[3]) > 0:
                    if row[0] in abst:
                        self.abstracts.append(abst[row[0]])
                        length = len(abst[row[0]].split())
                        if length > self.max_abs_length:
                            self.max_abs_length = length
                i += 1
        print('total number of labeled titles: ', len(self.titles))
        print('total number of labels: ', len(self.titles))
        return self.max_length
    
    def train_test_split(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

        for train_index, test_index in sss.split(self.titles, self.labels):
            self.titles_train, self.titles_test = np.array(self.titles)[train_index], np.array(self.titles)[test_index]
            self.labels_train, self.labels_test = np.array(self.labels)[train_index], np.array(self.labels)[test_index]
        print('train size:', len(self.titles_train) )
        print('test size:', len(self.titles_test))
        return self.titles_train, self.titles_test, self.labels_train, self.labels_test
