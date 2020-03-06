import tensorflow as tf

def linear_regression():
    '''
    # - 1.准备好数据集，我们制造y=0.8x+0.7的100个样本
    # - 2.建立线性模型，随机初始化w1和b1，y=wx+b，目标求出权重w和偏置b ==（比如我们随机一个w为0.2，b为0.7，和我们假设已经知道的0.8和0.7差很远，所以接下来要确定损失函数，然后梯度下降优化来找到损失最小的w和b）==
    # - 3.确定损失函数（预测值与真实值之间的误差），即均方误差
    # - 4.梯度下降优化损失：需要制定学习率（超参数）
    :return:
    '''

    # - 1.准备好数据集，我们制造y=0.8x+0.7的100个样本
    # 特征值x，目标值y_true
    x = tf.random_normal(shape=(100, 1), mean=2, stddev=2)
    y_true = tf.matmul(x, [[0.8]]) + 0.7

    ## 2.建立线性模型，目标：求出权重W和偏置b
    # y = W·X + b

    ## 3.随机初始化w1和b1
    w_ran = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)))
    b_ran = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)),trainable = True)#trainable = False就是大模型里面指定一些参数不被训练
    y_predict = tf.matmul(x, w_ran) + b_ran

    ## 4.确定损失函数（预测值与真实值之间的误差）即均方误差
    error = tf.reduce_mean(tf.square(y_predict - y_true))

    ## 5.梯度下降优化损失：需要制定学习率（超参数）
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 至此整张图就好了

    # 开启会话进行训练

    with tf.Session() as sess:
        # 运行初始化变量op
        sess.run(init)
        print("随机初始化的权重为%f， 偏置为%f" % (w_ran.eval(), b_ran.eval()))
        # 训练模型
        for i in range(100):
            sess.run(optimizer)
            print("第%d步的误差为%f，权重为%f， 偏置为%f" % (i, error.eval(), w_ran.eval(), b_ran.eval()))
if __name__ == '__main__':
    linear_regression()