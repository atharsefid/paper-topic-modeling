from __future__ import print_function
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from attention_model import *
#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input, Conv1D, MaxPooling1D, Flatten, Bidirectional, Dropout, Merge
from keras.layers import LSTM
from keras.models import Model
from data_reader import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras import backend as K
import numpy
numpy.random.seed(0)


# Misc Parameters
tf.flags.DEFINE_integer("EMBEDDING_DIM", 200, "embedding diemension (default:200)")
tf.flags.DEFINE_string("EMBEDDING_FILE", "pubmed_200.vec", "embedding file")
tf.flags.DEFINE_string("DATA_FILE", "title_labels.tsv", "file containing training data")
                                                                                                                                                                                   

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


EMBEDDING_DIM =FLAGS.EMBEDDING_DIM
reader = data()
MAX_SEQUENCE_LENGTH = reader.read_data(FLAGS.DATA_FILE)
MAX_ABS_SEQUENCE_LENGTH = reader.read_abstracts(FLAGS.DATA_FILE)
titles = reader.titles
abstracts = reader.abstracts
labels = reader.labels
titles.pop(0)
titles.pop(-1)
labels.pop(0)
labels.pop(-1)
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(titles + abstracts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# read embeddings
embeddings_index = {}
f = open('./wordEmbedding/'+FLAGS.EMBEDDING_FILE,'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# make embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Found %s word vectors.' % len(embeddings_index))

# title model
title_model = Sequential()
title_model.add(Embedding(len(word_index) + 1,                                                                
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
#model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, )))
title_model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, ))
title_model.add(Dropout(0.1))

# abstract model
abstract_model = Sequential()
abstract_model.add(Embedding(len(word_index) + 1,                                                                
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_ABS_SEQUENCE_LENGTH,
                            trainable=False))
#model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, )))
abstract_model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.3, ))
abstract_model.add(Dropout(0.5))

merged = Merge([title_model, abstract_model], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1, activation='sigmoid'))
final_model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=['acc'])


# train test split
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
p = []
r = []
for train_index, test_index in skf.split(titles, labels):
    titles_train = np.array(titles)[train_index]
    titles_test = np.array(titles)[test_index]
    abstract_train = np.array(abstract)[train_index]
    abstract_test = np.array(abstract)[test_index]
    labels_train = np.array(labels)[train_index]
    labels_test = np.array(labels)[test_index]
    train_sequences = tokenizer.texts_to_sequences(titles_train)
    abstract_train_sequences = tokenizer.texts_to_sequences(abstract_train)
    test_sequences = tokenizer.texts_to_sequences(titles_test)
    abstract_test_sequences = tokenizer.texts_to_sequences(abstract_test)

    train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    abstract_train = pad_sequences(abstract_sequences, maxlen=MAX_ABS_SEQUENCE_LENGTH)
    test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    abstract_test = pad_sequences(test_sequences, maxlen=MAX_ABS_SEQUENCE_LENGTH)

    print('Shape of train data tensor:', train.shape)
    print('Shape of test data tensor:', test.shape)
    print('Shape of label tensor:', labels_train.shape)

    # happy learning!
    model.fit([train, abstract_train], labels_train, epochs=50, batch_size=128)
    score, acc = model.evaluate([test, abstract_test ], labels_test, batch_size=1)
    pred = model.predict_classes([test,abstract_test], batch_size=1)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    p.append(precision)
    r.append(recall)
    print(' ')
    print('precision:', precision)
    print('recall:', recall)
    print('Test score:', score)
    print('Test accuracy:', acc)

total_p = round(sum(p)/len(p),3)                                         
total_r = round( sum(r)/len(r),3)
total_f1 = round(2 * total_p * total_r /(total_p+total_r),3) 
print('precision:{}, recall :{}, f1:{} '.format(total_p, total_r, total_f1))
