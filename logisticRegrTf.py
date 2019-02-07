import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

xData = np.linspace(0.0, 10.0, 100000)
noise = np.random.randn(len(xData))

print(xData)
print(noise.shape)

yTrue = (0.5)*xData + 5 + noise

xDf = pd.DataFrame(data=xData, columns=['x data'])
yDf = pd.DataFrame(data=yTrue, columns=['y'])

print(xDf.head())
print(yDf.head())

myData = pd.concat([xDf, yDf], axis=1)
print(myData.head())

myData.sample(n=250).plot(kind='scatter', x='x data', y='y')
plt.show()

batchSize = 1000

# random m and b
m = tf.Variable(0.1)
b = tf.Variable(0.14)

xph = tf.placeholder(tf.float32, [batchSize])
yph = tf.placeholder(tf.float32, [batchSize])

y_model = m*xph + b

error = tf.reduce_sum(tf.square(yph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        randIndex = np.random.randint(len(xData), size=batchSize)
        feed = {xph:xData[randIndex], yph:yTrue[randIndex]}
        #print(feed)
        sess.run(train, feed_dict=feed)
        model = sess.run([m, b])

print(model[0], model[1])

plt.plot(xData, model[0]*xData + model[1], 'r')
index = np.random.randint(len(xData), size=250)
print(index)
plt.scatter(xData[index], yTrue[index])
plt.show()
