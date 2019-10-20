import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class NeuralNetwork:
   def __init__(self):
       trainingData = np.load('../InputDataSL.npy')
       labels = np.load('../TrainingData.npy')
       labels = np.reshape(labels, [-1, 303])
       try:
           self.indexs = np.load('../shuffled_indexes.npy')
       except:
           self.indexs = np.random.permutation(np.arange(328))
       self.trainingData = trainingData[self.indexs, :]
       self.labels = labels[self.indexs, :]
       np.save('../shuffled_indexes.npy', self.indexs)
       self.x = tf.placeholder(tf.float32, shape=[None, 4])
       self.y = tf.placeholder(tf.float32, shape=[None, 303])
       self.loss, self.output = regressor(self.x, self.y)
       self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
       self.writer, self.summary_op = create_summaries(self.loss, self.x, self.output)
       self.saver = tf.train.Saver()
   def train(self):
     with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         try:
             self.saver.restore(sess, "../model2")
         except:
             pass
         for i in range(int(40800)):
             feed = {self.x : self.trainingData[:-30], self.y: self.labels[:-30]}
             feed_test = {self.x : self.trainingData[-30:], self.y: self.labels[-30:]}
             if i % 100 == 0:
                 summary, train_loss = sess.run([self.summary_op, self.loss], feed_dict = feed)
                 print("step %d, training loss: %g" % (i, train_loss))
                 test_loss = sess.run(loss, feed_dict = feed_test)
                 print("step %d, test loss: %g" % (i, test_loss))
                 self.writer.add_summary(summary, i)
                 self.writer.flush()
             self.train_step.run(feed_dict=feed)   
         self.saver.save(sess, "../model2")
      
  def predict(self, ip):
     with tf.Session() as sess:
         self.saver.restore(sess, "../model2")
         op = sess.run(self.output, feed_dict={self.x : ip})
     return op 
  def weight_variable(shape):
     initial = tf.truncated_normal(shape, stddev=0.1)
     return tf.Variable(initial)
  def bias_variable(shape):
     initial = tf.constant(0.1, shape=shape)
     return tf.Variable(initial)
  def fc_layer(previous, input_size, output_size):
     W = weight_variable([input_size, output_size])
     b = bias_variable([output_size])
     return tf.matmul(previous, W) + b
  def regressor(x, y):
     l1 = tf.nn.tanh(fc_layer(x, 4, 12))
     l2 = fc_layer(l1, 12, 303)
     loss = tf.reduce_mean(tf.squared_difference(y,l2))
     return loss, l2
  def create_summaries(loss, x, output):
     writer = tf.summary.FileWriter("../logs")
     tf.summary.scalar("Loss", loss)
     return writer, tf.summary.merge_all()
