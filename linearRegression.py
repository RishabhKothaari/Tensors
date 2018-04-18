import numpy
import torch
from sklearn import datasets

data = numpy.genfromtxt('./data/data.txt',delimiter=',')
x = numpy.zeros((data.shape))
x[:,0] = data[:,0]
x[:,1] = 1.0
xTensor = torch.from_numpy(x)
y = numpy.zeros((data.shape[0],1))
y[:,0] = data[:,1]
yTensor = torch.from_numpy(y)

alpha,_ = torch.gels(yTensor,xTensor)
