import numpy as np
# import tensorflow as tf
import torch
import pdb

import q_learning

# class SGDRegressor:
#     def __init__(self, D):
#         print("Ver.Tensorflow")
#         lr = 0.1

#         # create inputs, targets, params
#         # matmul doesn't like when w is 1-D
#         # so we make it 2-D and then flatten the prediction
#         self.w = tf.Variable(tf.random_norman(shape=(D,1)), name='w')
#         self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
#         self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

#         # make prediction and cost
#         Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1]) # invert it into 1d tensor
#         delta = self.Y - Y_hat
#         cost = tf.reduce_sum(delta * delta)

#         # ops we want to call later
#         self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
#         self.predict_op = Y_hat

#         # start the session and initialize params
#         init = tf.global_variables_initializer()
#         self.session = tf.InteractiveSession()
#         self.session.run(init)

#     def partial_fit(self, X, Y):
#         self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

# class SGDRegressor: # tensorflow 2.0 version
#     def __init__(self, D, lr=1e-2):
#         self.w = tf.Variable(tf.random.normal((D, 1)), name='w')
#         self.b = tf.Variable(tf.zeros(1, dtype=tf.float32), name='b')
#         self.lr = lr

#     def partial_fit(self, X, Y):
#         with tf.GradientTape(persistent=True) as tape:
#             y_hat = X @ self.w + self.b
#             loss = tf.reduce_mean(tf.square(Y - y_hat))
#         dw, db = tape.gradient(loss, [self.w, self.b])
 
#         self.w.assign_sub(self.lr * dw)
#         self.b.assign_sub(self.lr * db)
 
#     def predict(self, X):
#         return tf.reshape(X @ self.w + self.b, [-1])

class SGDRegressor: # pytorch version
    def __init__(self, D, lr=1e-2):
        self.w = torch.randn((D,1), requires_grad = True)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad = True)
        self.lr = lr

    def partial_fit(self, X, Y):
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        y_hat = torch.matmul(X, self.w) + self.b
        loss = torch.mean(torch.square(Y - y_hat))

        loss.backward()

        dw, db = self.w.grad, self.b.grad
        with torch.no_grad():
            self.w -= self.lr * dw
            self.b -= self.lr * db

        self.w.grad.zero_()
        self.b.grad.zero_()
 
    def predict(self, X):
        X = torch.Tensor(X)
        return (torch.reshape(torch.matmul(X, self.w) + self.b, [-1])).detach().numpy()


if __name__ == "__main__":
    q_learning.SGDRegressor = SGDRegressor
    q_learning.main()


