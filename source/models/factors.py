import torch
import torch.nn as nn


class SumAlong(nn.Module):
    """
    Takes the sum of two variables using indices
    """
    def __init__(self):
        super(SumAlong, self).__init__()

    def forward(self, x0, x1, i0, i1):
        """
        Return x0[i0] + x1[i1]

        :param x0:
        :param x1:
        :param i0:
        :param i1:
        :return:
        """
        return x0[i0.reshape((-1,)), :] + x1[i1.reshape((-1,)), :]


class InnerProduct(nn.Module):
    """
    Takes the row-wise inner product between two matrices
    """
    def __init__(self):
        super(InnerProduct, self).__init__()

    def forward(self, x, y):
        """
        Return xi'yi for all i

        :param x:
        :param y:
        :return:
        """
        return torch.sum(x * y, 1, keepdim=True)


class Distance(nn.Module):
    """
    Takes the row-wise L2 squared distance
    """
    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, x, y):
        """
        Return xi'yi for all i

        :param x:
        :param y:
        :return:
        """
        return torch.sum((x - y)**2, 1, keepdim=True)
