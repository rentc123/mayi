import math
class CLR():
    def __init__(self, data_size, times, batch_size, min_rate, max_rate):
        """
        构造函数
        :param data_size: 数据集大小
        :param times:   倍数[2-8] step_size=one_epoch_steps*times
       :param batch_size: 批大小
        :param max_rate: 最大学习速率
        :param min_rate: 最小学习速率
        """
        self.max_rate = max_rate
        self.min_rate = min_rate
        one_epoch_steps = data_size // batch_size
        self.step_size = one_epoch_steps * times

    def getlr(self, steps):
        """
        获取当前的学习速率
        :param steps: 当前的总迭代次数。
        :return: 学习速率
        """
        cycle = math.floor(1 + steps / (2 * self.step_size))
        x = abs(steps / self.step_size - 2 * cycle + 1)
        lr = self.min_rate + (self.max_rate - self.min_rate) * max(0, (1 - x))
        return lr

    def lrtest(self, lr):
        """
        可以使用低学习速率训练一些轮数，每一批训练的时候逐渐提高（指数或线性）学习
        速率，画出图形来观察最大最小学习速率在哪个区间。
        :param lr:
        :return:
        """
        return
