import torch
import numpy
import time

print("torch version",torch.__version__)

print("#1 ")

x = torch.Tensor(13,13).fill_(1)
x.narrow(0,6,1).fill_(2)
x.narrow(1,6,1).fill_(2)
x.narrow(1,1,1).fill_(2)
x.narrow(1,11,1).fill_(2)
x.narrow(0,1,1).fill_(2)
x.narrow(0,11,1).fill_(2)
x.narrow(1,3,1)[3].fill_(3)
x.narrow(1,3,1)[4].fill_(3)
x.narrow(1,4,1)[3].fill_(3)
x.narrow(1,4,1)[4].fill_(3)
x.narrow(1,3,1)[8].fill_(3)
x.narrow(1,3,1)[9].fill_(3)
x.narrow(1,4,1)[8].fill_(3)
x.narrow(1,4,1)[9].fill_(3)
x.narrow(1,8,1)[3].fill_(3)
x.narrow(1,8,1)[4].fill_(3)
x.narrow(1,9,1)[3].fill_(3)
x.narrow(1,9,1)[4].fill_(3)
x.narrow(1,8,1)[8].fill_(3)
x.narrow(1,8,1)[9].fill_(3)
x.narrow(1,9,1)[8].fill_(3)
x.narrow(1,9,1)[9].fill_(3)

print(x)

print("Eigendecomposition")
matrix = torch.Tensor(20,20)
a = torch.diag(torch.arange(1,matrix.size(0)+1))
result = a.mm(matrix).mm(a.inverse())
vectors,values = result.eig()
print("values",vectors.narrow(1,0,1).squeeze().sort()[0])
print("values",values)

print("flops per second")

a = torch.Tensor(5000,5000).normal_()
b = torch.Tensor(5000,5000).normal_()
t1 = time.perf_counter()
c = a.mm(b)
t2 = time.perf_counter()
print("flops per second - ",(5000*5000*5000)/(t2-t1))


def mul_row(a):
    result = a
    for i in range(0,a.size(0)):
        for j in range(0,a.size(1)):
            result[i,j] = result[i,j]*(i+1)
        #end
    #end
    return result

def mul_row_fast(a):
    mul = torch.arange(1,a.size(0)+1).view(a.size(0),1).expand_as(a)
    return torch.mul(a,mul)

matrix = torch.Tensor(10000,400).fill_(2.0)
t1 = time.perf_counter()
result1 = mul_row(matrix)
t2 = time.perf_counter()
print("time 1",t2-t1)
t3 = time.perf_counter()
result2 = mul_row_fast(matrix)
t4 = time.perf_counter()
print("time 2",t4-t3)
print("checking - ",torch.norm(result1-result2))