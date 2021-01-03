import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

## 下载数据集
(x_train,y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()
# print(x_train.shape,x_test.shape)

"""
预处理
"""
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32) / 255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y

## 参数设置
batchsz = 128
## 优化梯度下降
optimizer = optimizers.Adam(lr = 1e-3)

## 整理数据集
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)) 
db_train = db_train.map(preprocess).shuffle(10000).batch(batchsz)
# print(db_train)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsz)
# print(db_test)

## 5层网络结构
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape = [None,28*28])  ## 初始化参数
model.summary()

## 主函数
def mian ():
    total_correct , total_sum = 0 , 0 ## 计算参数
    for epoch in range(10):
        for step , (x,y) in enumerate(db_train):
            x = tf.reshape(x,[-1,28*28])
            ## 梯度下降
            with tf.GradientTape() as tape:
                logits = model(x) # 输出
                y_onehot = tf.one_hot(y,depth = 10) 
                ## 计算损失函数
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True))
            ## 更新权值
            grads = tape.gradient(loss_ce,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            if step % 100 == 0:
                print(epoch,step,"loss:",float(loss_ce))

        ## 测试训练结果
        for x , y  in db_test:
            # print(y)
            x = tf.reshape(x,[-1,28*28]) ## 转换格式
            logits = model(x)
            # print(logits) [b,10]
            prob = tf.nn.softmax(logits,axis = 1) ## 归一化
            # print(prob) [b,10]
            pred = tf.cast(tf.argmax(prob,axis = 1) ,tf.int32)## 得到索引
            # print(pred)
            correct = tf.cast(tf.equal(y,pred),dtype = tf.int32)  ## 计算个数
            # print(correct)
            correct =tf.reduce_sum(correct)
            # print(correct)
            total_correct += correct  ## 计算总数
            total_sum += y.shape[0] 
            # print(y.shape[0])
        acc  = total_correct / total_sum
        print(epoch,"acc:",acc)
        
## 开始
if __name__ == "__main__":
    mian( )