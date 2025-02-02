import numpy as np
#全连接层实现类
class FullConnectedLayer():
    def __init__(self, input_size, output_size, activator):
        """
        构造函数
        :param input_size:本层输入向量的维度
        :param output_size:本层输出向量的维度
        :param activator:激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        #权重数组W
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size,1))
        self.output = np.zeros((output_size,1))
    def forward(self, input_array):
        """
        正向传播
        :param input_array:输入向量，维度需要等于input_size
        :return:
        """
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)
    def backward(self, delta_array):
        """
        反向传播求W和b
        :param delta_array:上一层传来的误差项
        :return:
        """
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array
    def update(self, learning_rate):
        """
        梯度下降更新权重
        :param learning_rate:学习率
        :return:
        """
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

# Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        return output * (1 - output)
# 神经网络类
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''

        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )
    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本x
        '''

        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output
    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                    data_set[d], rate)
    def train_one_sample(self, label, sample, rate):
        sample = sample.reshape(-1, 1)
        label =label.reshape(-1, 1)
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)
    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta
    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

