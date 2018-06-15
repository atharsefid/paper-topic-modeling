from __future__ import print_function
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from attention_model import *
#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input, Conv1D, MaxPooling1D, Flatten, Bidirectional, Dropout
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


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


EMBEDDING_DIM =FLAGS.EMBEDDING_DIM
reader = data()
MAX_SEQUENCE_LENGTH = reader.read_data(FLAGS.DATA_FILE)
titles = reader.titles
labels = reader.labels

tokenizer = Tokenizer() 
tokenizer.fit_on_texts(titles)
sequences = tokenizer.texts_to_sequences(titles)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# read embeddings
embeddings_index = {}
f = open('wordEmbedding/'+FLAGS.EMBEDDING_FILE,'r')
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

# build model
model = Sequential()
model.add(Embedding(len(word_index) + 1,                                                                
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
#model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, )))
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, ))
model.add(Dropout(0.1))
#model.add(attention_3d_block)
#model.add(AttentionDecoder(128, EMBEDDING_DIM))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=['acc'])
'''
train_sequences = tokenizer.texts_to_sequences(titles)
test_sequences = tokenizer.texts_to_sequences(test_titles)

train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of train data tensor:', train.shape)
print('Shape of test data tensor:', test.shape)

# happy learning!
model.fit(train, labels, epochs=50, batch_size=128)
pred = model.predict_classes(test, batch_size=1)
with open('prediction.txt','w') as f:
    for a,b in zip(test_titles, pred):
        print(a , b)
        f.write(a + '\t' + str(b) + '\n')
'''
# train test split
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
p = []
r = []
for train_index, test_index in skf.split(titles, labels):
    print('train set size:', len(train_index))
    print('test set size:', len(test_index))
    titles_train = np.array(titles)[train_index]
    titles_test = np.array(titles)[test_index]
    labels_train = np.array(labels)[train_index]
    labels_test = np.array(labels)[test_index]
    train_sequences = tokenizer.texts_to_sequences(titles_train)
    test_sequences = tokenizer.texts_to_sequences(titles_test)

    train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of train data tensor:', train.shape)
    print('Shape of test data tensor:', test.shape)
    print('Shape of label tensor:', labels_train.shape)

    # happy learning!
    model.fit(train, labels_train, epochs=50, batch_size=128)
    score, acc = model.evaluate(test, labels_test, batch_size=1)
    pred = model.predict_classes(test, batch_size=1)
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
