import sys
sys.path.append(r'C:\Users\ppdash\workspace\Do-It-Yourslef-data-science')
#Load package
from common.data_server import BatchDataServer
from common.h5_reader_writer import H5Writer
from da_vd_credit import build_vgg16 as vgg16
import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


#class to detect defaulter from credi card dataset
class credit_card_data:
    def __init__(self):
        df = pd.read_csv("UCI_Credit_Card.csv")
        df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)
        df.loc[df.default == 0, 'nonDefault'] = 1
        df.loc[df.default == 1, 'nonDefault'] = 0


        Default = df[df.default == 1]
        NonDefault = df[df.nonDefault == 1]

        # Set X_train equal to 80% of the observations that defaulted.
        X_train = Default.sample(frac=0.8)
        count_Defaults = len(X_train)

        # Add 80% of the not-defaulted observations to X_train.
        X_train = pd.concat([X_train, NonDefault.sample(frac=0.8)], axis=0)

        # X_test contains all the observations not in X_train.
        X_test = df.loc[~df.index.isin(X_train.index)]

        # Shuffle the dataframes so that the training is done in a random order.
        X_train = shuffle(X_train)
        X_test = shuffle(X_test)

        # Add our target classes to y_train and y_test.
        y_train = X_train.default
        y_train = pd.concat([y_train, X_train.nonDefault], axis=1)

        y_test = X_test.default
        y_test = pd.concat([y_test, X_test.nonDefault], axis=1)

        # Drop target classes from X_train and X_test.
        X_train = X_train.drop(['default', 'nonDefault'], axis=1)
        X_test = X_test.drop(['default', 'nonDefault'], axis=1)

        # Check to ensure all of the training/testing dataframes are of the correct length
        print(len(X_train))
        print(len(y_train))
        print(len(X_test))
        print(len(y_test))

        features = X_train.columns.values


        for feature in features:
            mean, std = df[feature].mean(), df[feature].std()
            #X_train.loc[:, feature] = X_train[feature]
            #X_test.loc[:, feature] = X_test[feature]

            X_train.loc[:, feature] = (X_train[feature] - mean) / std
            X_test.loc[:, feature] = (X_test[feature] - mean) / std

        # Split the testing data into validation and testing sets
        split = int(len(y_test) / 2)

        self.inputX = X_train.as_matrix()
        self.inputY = y_train.as_matrix()
        self.inputX_valid = X_test.as_matrix()[:split]
        self.inputY_valid = y_test.as_matrix()[:split]
        self.inputX_test = X_test.as_matrix()[split:]
        self.inputY_test = y_test.as_matrix()[split:]



class DeepNILM:
    """
     DeepNILM class is designed to claculate weight (w) and bias (b) from a set of inputs and matching outputs.

     Args:
         lr  is learning rate
         num_epochs
         print_frequency


          vgg16 is the class with convolution and fully connected layers

    """
    def __init__(self, config):
        self.config = config
        self.lr = config['learning_rate']
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.n_samples = None
        self.num_epochs = config['num_epochs']
        self.print_frequency = config['print_frequency']
        self.feature_dimension = 24
        self.number_class = 2

        self.Y = tf.placeholder(shape=(None, self.number_class), dtype=tf.float32)
        vgg_graph = vgg16(init_node=tf.placeholder(tf.float32, [None, 1,  self.feature_dimension, 1]))
        self.X  = vgg_graph['tn']['place_holder']
        self.tensors = vgg_graph['tn']
        self.variables = vgg_graph['var']
        self.model()

    def model(self):
        self.tensors['y_hat'] = tf.nn.softmax(self.tensors['fc_3'])
        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.predictor(), 1))
        self.tensors['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predictor(self):
        return self.tensors['y_hat']

    def predict(self, learned_params, X=None):
        self.model();
        with tf.Session() as sess:
            self.update_tensors_with_learned_params(learned_params, sess)
            sess.run(tf.global_variables_initializer())
            return sess.run([self.predictor()],
                            feed_dict={self.X: X})

    def compute_stats(self, x, y, sess):
        batch_data = BatchDataServer(x, y, batch_size=128)
        total_accuracy = 0
        confusion_tensor = tf.stack([tf.argmax(self.Y, 1), tf.argmax(self.predictor(), 1)])
        confusion_mat = np.zeros([self.number_class, self.number_class], int)

        while batch_data.epoch < len(batch_data):
            x1, y1 = batch_data.next()

            [accuracy, con_mat] = sess.run([self.tensors['accuracy'], confusion_tensor],
                                       feed_dict={self.X: x1, self.Y: y1})
            total_accuracy = total_accuracy + accuracy
            for p in con_mat.T:
                confusion_mat[p[0], p[1]] += 1

        dataframe = pd.DataFrame(confusion_mat)
        label = ['df', 'ndf']
        dataframe.index = label
        dataframe.columns = label

        print(pd.DataFrame(dataframe))

        return total_accuracy / len(batch_data), pd.DataFrame(dataframe)

    def update_tensors_with_learned_params(self, learned_params, sess):
        for key in learned_params:
            sess.run(self.variables[key].assign(learned_params[key]))

    def cost_function(self):
        return tf.reduce_mean(-tf.reduce_sum( self.Y * tf.log(self.predictor()+ 1e-10), reduction_indices=[1]))

    def solver(self):
        return tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(self.cost_function())


    def get_trained_params(self, sess):
        return self.variables


    def write_params_to_file(self, filename, params_dict):
        H5Writer.write(filename, self.config, params_dict)

    def train(self, x=None, y=None, x_test=None, y_test=None, filename=None):
        cost_function = self.cost_function()
        solver = self.solver();
        batch_data = BatchDataServer(x, y, batch_size = 128)
        lr = self.lr
        loss = []
        accuracy_training =[]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            accuracy, conf_mat = self.compute_stats(x, y, sess)
            print("accuracy on testing appliance data before training are", accuracy)
            while batch_data.epoch < self.num_epochs:
                x1, y1 = batch_data.next()
                [_, cost, accuracy] = sess.run([solver, cost_function, self.tensors['accuracy']],
                                         feed_dict={self.X: x1, self.Y: y1,  self.learning_rate:lr})
                loss.append(cost)
                accuracy_training.append(accuracy)

                if (batch_data.epoch + 1) % self.print_frequency == 0:
                    print('At Epoch {} the loss is {}'.format(batch_data.epoch, cost))

                    batch_data.epoch = batch_data.epoch + 1
                    lr = self.lr * numpy.exp(-3.1*(batch_data.epoch/(self.num_epochs+1.0)))
                    print(lr)
                    params_dict = self.get_trained_params(sess)
                    accuracy, frame = self.compute_stats(x, y, sess)
                    self.config['learning_rate'] = lr
                    print("accuracy on testing appliance data after training is {} with a learning rate of {}".format(accuracy, lr))
                    frame = frame / frame.sum()
                    if accuracy > 0.7:
                        print('writing to')
                        frame.to_csv('confusion.csv')
                #if (batch_data.epoch + 1) % 10000 == 0:
                    print("performing testing \n \n \n ...................................")
                    accuracy_test, conf_mat_test = self.compute_stats(x_test, y_test, sess)
                    # print("testing _confmat", conf_mat_test)
                    print("Testing accuracy is {} with a learning rate of {}".format(accuracy_test, lr))

        return_dict = {'parameters': params_dict, 'loss': loss, 'Accuracy': accuracy_training}
        #self.write_params_to_file('nilm_vd16.da', self.variables)
        return return_dict





import numpy as np
import random




def test_train_credit_data():
    credit_data = credit_card_data()

    config = {
        'num_hidden_layers': 5,
        'order_poly': 4,
        'learning_rate': 0.0001,
        'num_epochs': 20000,
        'print_frequency': 1000,
    }

    lr = DeepNILM(config)
    train_x = credit_data.inputX
    train_x = train_x.reshape((-1, 1, 24, 1))
   # train_x = train_x / np.max(train_x) - 0.5
    train_y = credit_data.inputY
    test_x = credit_data.inputX_valid
    #test_x = test_x / np.max(test_x) - 0.5
    test_x = test_x.reshape((-1, 1, 24, 1))

    test_y = credit_data.inputY_valid
    learned_params = lr.train(x=train_x, y=train_y, x_test=test_x, y_test=test_y, filename=None)


    loss = learned_params['loss']
    accuracy_training = learned_params['Accuracy']
    plt.plot(loss, color='k')
    plt.xlabel('epoch', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.show()
    plt.plot(accuracy_training, linewidth=3, color='k')
    plt.xlabel('epoch', fontsize=30)
    plt.ylabel('Accuracy', fontsize=30)
    plt.show()


if __name__ == '__main__':

    test_train_credit_data()


