import numpy as np
import tensorflow as tf

class tf_2layer_nn(object):
    '''
    The two_layered_nn contains all the logic required to implement a two layered neural network.
    '''
    def __init__(self,data,**kwargs):
        '''
        Create an instance to initialize all variables
        '''
        self.X_train = data['train_']
        self.Y_train = data['tr_labels']
        self.X_val = data['val']
        self.Y_val = data['val_labels']
        
        self.classes = kwargs.pop('classes',10)
        self.batch_size = kwargs.pop('batch_size', 200)
        self.num_iterations = kwargs.pop('num_iterations',100)
        self.hidden_size = kwargs.pop('hidden_size',78)

        # Activation types
        self.activation_type = kwargs.pop('activation_type', 'relu')
        
        # general parameters
        self.reg_type = kwargs.pop('reg_type','None')
        self.drop = kwargs.pop('dropout', True)
        self.learning_rate = kwargs.pop('learning_rate', 7.5e-4)
        self.reg = kwargs.pop('reg',0.1)
        self.delta = kwargs.pop('delta', 1e-8)
        self.keep_prob = kwargs.pop('keep_prob',0.5)
        # for ADAM
        self.beta1 = kwargs.pop('beta1', 0.9)
        self.beta2 = kwargs.pop('beta2', 0.999)
        #for MOMENTUM
        self.momentum = kwargs.pop('momentum', 0.9)
        # for RMSPROP
        self.decay_rate = kwargs.pop('decay_rate',0.95)
        
        # set default optimization type to sgd
        self.optimization_type = kwargs.pop('optimization_type','SGD')
        
        # Setting the input and output variables
        self.x = tf.placeholder(tf.float32, shape=(None,self.X_train.shape[1]), name = 'X')
        self.y = tf.placeholder(tf.float32,shape=(None,self.classes), name = 'Y')
        

        self.params = {}
        self.params['w1'] = tf.Variable(tf.random_normal([self.X_train.shape[1],self.hidden_size]), name = 'w1')
        self.params['b1'] = tf.Variable(tf.random_normal([self.hidden_size]), name = 'b1')
        self.params['w2'] = tf.Variable(tf.random_normal([self.hidden_size,self.classes]), name = 'w2')
        self.params['b2'] = tf.Variable(tf.random_normal([self.classes]),name = 'b2')
        
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.val_score = []
        self.print_stats = kwargs.pop('print_stats', True) # enable this if you want updates on val and train acc every 10 iters
        


    def predict(self, X_batch):
        # The First layer 
        input_1 = tf.add(tf.matmul(X_batch, self.params['w1']), self.params['b1'])

        # activation selection
        if self.activation_type is 'relu': 
            input_2 = tf.nn.relu(input_1) 
        elif self.activation_type is 'tanh':
            input_2 = tf.nn.tanh( input_1)
        elif self.activation_type is 'sig':
            input_2 = tf.nn.sigmoid(input_1)
        else:
            input_2 = tf.nn.relu(input_1) 

        # dropout selection
        if self.drop is True:
           output = tf.nn.dropout(input_2, self.keep_prob)
        
        # The Second layer
        scores = tf.add(tf.matmul(output, self.params['w2']), self.params['b2'])

        return (scores)


    def train(self):


        if self.reg_type is 'L2':
            scores = self.predict(self.x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = self.y))
            reg_cost = self.reg * (tf.nn.l2_loss(self.params['w1']) + tf.nn.l2_loss(self.params['w2']))
            cost = cost + reg_cost
        else:
            scores = self.predict(self.x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = self.y))
            
        correct = tf.equal(tf.argmax(scores,1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float')) * 100  # gives accuracy in percentages

        if self.optimization_type is 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        elif self.optimization_type is 'ADAM':
            optimizer = tf.train.AdamOptimizer(self.learning_rate,self.beta1,self.beta2,self.delta).minimize(cost)
        elif self.optimization_type is 'MOM':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.momentum).minimize(cost)
        elif self.optimization_type is 'RMSPROP':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate,self.decay_rate).minimize(cost)
        
        # Running a Tensorflow session to run the whole 2 layer NN tensorflow computational graph 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for it in range(self.num_iterations):
                
                # generate random batches
                idx = np.random.choice(np.int(self.X_train.shape[0]), np.int(self.batch_size), replace = True)
                ex = self.X_train[idx]
                ey = self.Y_train[idx]
                
                _,c,acc = sess.run([optimizer,cost,accuracy], feed_dict = {self.x:ex,self.y:ey})
                val_c,val_acc = sess.run([cost,accuracy], feed_dict = {self.x:self.X_val, self.y:self.Y_val})

                if self.print_stats == True:
                    if it % 10 == 0:
                        print ('iteration '+str(it) + ' / '+ str(self.num_iterations) +' :loss ' + str(c))
                        print('training accuracy: '+ str(acc) + ' and validation accuracy: '+ str(val_acc))

                self.train_loss_history.append(c)
                self.train_acc_history.append(acc)

                self.val_loss_history.append(val_c)
                self.val_acc_history.append(val_acc)

            self.val_score = sess.run([scores], feed_dict = {self.x:self.X_val, self.y:self.Y_val})
            #self.val_score = self.val_score.reshape(self.val_score.shape[1:])

                
