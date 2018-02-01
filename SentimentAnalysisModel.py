'''
Extract files and convert them to features.
'''
import numpy as np
import tensorflow as tf
import FeatureExtractor as fe


class SentimentAnalysisModel:
    '''
    stuff
    '''
    def __init__(self, featureExtractor, seq_length):
        self.max_sequence_length = seq_length
        self.FeatureExtractor = featureExtractor
        self.split_frac = 0.8

        self.lstm_size = 256   #Number of units in the hidden layers in the LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
        self.lstm_layers = 1  #Number of LSTM layers in the network. I'd start with 1, then add more if I'm underfitting.
        self.batch_size = 500 #The number of reviews to feed the network in one training pass. Typically this should be set as high as you can go without running out of memory.
        self.learning_rate = 0.001  #Learning rate
        self.embed_size = 300
        self.__tfGraph = 1

    def Build_Graph(self):
        '''
        Build the magic here
        '''
        # Adding 1 because we use 0's for padding, dictionary started at 1
        n_words = len(self.FeatureExtractor.TrainingData_Vocab_To_Int) + 1

        # Add nodes to the graph
        inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
        labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        embedding = tf.Variable(tf.random_uniform((n_words, self.embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        # Your basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * self.lstm_layers)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        #Output layer
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
        cost = tf.losses.mean_squared_error(labels_, predictions)

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        return inputs_, labels_, keep_prob, cell, initial_state, final_state, cost, optimizer, accuracy


    def get_batches(self, x, y, batch_size=100):    
        n_batches = len(x)//batch_size
        x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
        for ii in range(0, len(x), batch_size):
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]



    def Train2(self):

        tf.reset_default_graph()
        graph = tf.Graph()
        
        with graph.as_default():
            self.Train()

    def Train(self):

        '''
        Train our model
        '''
        self.FeatureExtractor.ExtractFeatures()
        features = self.FeatureExtractor.Features
        labels =  self.FeatureExtractor.ExpectedOutputs

        split_idx = int(len(features) * self.split_frac)
        train_x, val_x = features[:split_idx], features[split_idx:]
        train_y, val_y = labels[:split_idx], labels[split_idx:]

        test_idx = int(len(val_x)*0.5)
        val_x, test_x = val_x[:test_idx], val_x[test_idx:]
        val_y, test_y = val_y[:test_idx], val_y[test_idx:]

        print("\t\t\tFeature Shapes:")
        print("Train set: \t\t{}".format(train_x.shape), 
            "\nValidation set: \t{}".format(val_x.shape),
            "\nTest set: \t\t{}".format(test_x.shape))



   


        inputs_, labels_, keep_prob, cell, initial_state, final_state, cost, optimizer, accuracy = self.Build_Graph()

        epochs = 10

        saver = tf.train.Saver()


        test_acc = []

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            test_state = sess.run(cell.zero_state(self.batch_size, tf.float32))
            for ii, (x, y) in enumerate(self.get_batches(test_x, test_y, self.batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 1,
                        initial_state: test_state}
                batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.3f}".format(np.mean(test_acc)))



        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            for e in range(epochs):
                state = sess.run(initial_state)
                
                for ii, (x, y) in enumerate(self.get_batches(train_x, train_y, self.batch_size), 1):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 0.5,
                            initial_state: state}
                    loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
                    
                    if iteration%5==0:
                        print("Epoch: {}/{}".format(e, epochs),
                            "Iteration: {}".format(iteration),
                            "Train loss: {:.3f}".format(loss))

                    if iteration%25==0:
                        val_acc = []
                        val_state = sess.run(cell.zero_state(self.batch_size, tf.float32))
                        for x, y in self.get_batches(val_x, val_y, self.batch_size):
                            feed = {inputs_: x,
                                    labels_: y[:, None],
                                    keep_prob: 1,
                                    initial_state: val_state}
                            batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                            val_acc.append(batch_acc)
                        print("Val acc: {:.3f}".format(np.mean(val_acc)))
                    iteration +=1
            saver.save(sess, "checkpoints/sentiment.ckpt")


   