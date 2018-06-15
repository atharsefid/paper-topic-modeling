from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from data_reader import *

def cross_validate(data, labels, clf):
    # train test split
    skf = StratifiedKFold(n_splits=10)
    p = []
    r = []
    for train_index, test_index in skf.split(data, labels):
        train = np.array(data)[train_index]
        test = np.array(data)[test_index]
        labels_train = np.array(labels)[train_index]
        labels_test = np.array(labels)[test_index]
        #print('Shape of train data tensor:', train.shape)
        #print('Shape of test data tensor:', test.shape)
        #print('Shape of label tensor:', labels_train.shape)

        # happy learning!
        clf.fit(train, labels_train)
        pred =clf.predict(test)
        precision = precision_score(labels_test, pred)
        recall = recall_score(labels_test, pred)
        p.append(precision)
        r.append(recall)
        #print('precision:', precision)
        #print('recall:', recall)
    total_p = round(sum(p)/len(p),3)
    total_r = round( sum(r)/len(r),3)
    total_f1 = round(2 * total_p * total_r /(total_p+total_r),3) 
    print('precision:{}, recall :{}, f1:{} '.format(total_p, total_r, total_f1))
    print("**************************************************************************")
DATA_FILE = "title_labels.tsv" 

reader = data()
MAX_SEQUENCE_LENGTH = reader.read_data(DATA_FILE)
titles = reader.titles
labels = reader.labels

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()), ])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (0.1, 0.01, 0.001, 0.0001, 0.00001),}

print('naive bayes results')
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
cross_validate(titles, labels, gs_clf)


#print('best score:', gs_clf.best_score_)
#print('best param:', gs_clf.best_params_)

# *********************************************apply stemming on data
'''import nltk
#nltk.download()
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                     ('tfidf', TfidfTransformer()),
                     ('mnb', MultinomialNB(fit_prior=False)),])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'mnb__alpha': (0.1, 0.01, 0.001, 0.0001, 0.00001),}

print('naive bayes results after stemming')
gs_mnb_clf = GridSearchCV(text_mnb_stemmed, parameters, n_jobs=-1)
cross_validate(titles, labels, gs_mnb_clf)
'''
################################################################################################################################################
# test on uptake
t, l  = reader.read_test('uptake.tsv')
gs_clf.fit(np.array(titles), np.array(labels))
pred = gs_clf.predict(np.array(t))

precision = precision_score(np.array(l), pred)
recall = recall_score(np.array(l), pred)
print(round(precision,3), round(recall,3) , round(2* precision * recall / (precision + recall),3))
