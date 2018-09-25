# Import the libraries
import tensorflow as tf
import numpy as np

# Define the Neural Accumulator Cell
def nac_cell(in_dim, out_dim):
    
    # shape[0]
    in_features = in_dim.shape[1]
    
    # Initialize weights variables
    W_hat = tf.get_variable(name='W_hat', initializer=tf.initializers.random_uniform(minval=2, maxval=2),
                            shape=[in_features, out_dim], trainable=True)
    M_hat = tf.get_variable(name='M_hat', initializer=tf.initializers.random_uniform(minval=2, maxval=2),
                            shape=[in_features, out_dim], trainable=True)
    
    # Define weight vector
    W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)
    
    # Define activations
    a = tf.matmul(in_dim, W)
    
    # Return activations and weights
    return a, W

# Generate training data
x1 = np.arange(0, 10000, 5, dtype=np.float32)
x2 = np.arange(5, 10005, 5, dtype=np.float32)

y_train = x1 + x2
x_train = np.column_stack((x1, x2))

# Generate test data
x3 = np.arange(11000, 20000, 8, dtype=np.float32)
x4 = np.arange(11000, 20000, 8, dtype=np.float32)

y_test = x3 + x4
x_test = np.column_stack((x3, x4))

# Define placeholders for input features and output vector
X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, ])

# Define a NAC layers
y_pred, _ = nac_cell(X, 1)
y_pred = tf.squeeze(y_pred)

# Define loss as mean squared error
loss = tf.reduce_mean((y_pred - Y) ** 2)

# Hyperparameters
lr = 0.005
epochs = 25000

# Define adam optimizer
optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Training
with tf.Session() as sess:
    cost_history = []
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(epochs+1):
        _, cost = sess.run([optimize, loss], feed_dict={X: x_train, Y: y_train})
        print("epoch: {}, MSE: {}".format(i, cost))
        cost_history.append(cost)
        
    # Print predicted output after training
    print("Predicted sum: ", sess.run(tf.ceil(y_pred[0:10]), feed_dict={X: x_test, Y: y_test}))