from perceptron import Perceptron
f = lambda x: x
class linearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数
        '''
        Perceptron.__init__(self, input_num, f)
def get_training_dataset():
    '''
    编造5个人的收入数据
    '''
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs,labels
def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    #创建神经元，输入参数的特征数是1（工作年长）
    lu = linearUnit(1)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu

if  __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
