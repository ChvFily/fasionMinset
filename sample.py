import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

## 下载数据集
(x_train,y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()
# print(x_train.shape,x_test.shape)

def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32) / 255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y


batchsz = 128

## 整理数据集
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)) 
db_train = db_train.map(preprocess).shuffle(10000).batch(batchsz)


# print(db_train)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsz)
# print(db_test)
## 5层网络结构




model = Sequential([
        layers.Dense(256, activation=tf.nn.relu)
        layers.Dense(128, activation=tf.nn.relu)
        layers.Dense(64, activation=tf.nn.relu)
        layers.Dense(32, activation=tf.nn.relu)
        layers.Dense(10)
])
model.build(input_shape = [None,28*28])  ## 初始化参数
model.summary()



 


## 网络




## 梯度下降

## 测试

