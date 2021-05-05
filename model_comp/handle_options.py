import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np



a = np.array([[1,2,4,6,8,4,5],[4,3,7,8,3,7,2]])
#a = np.array([0,1,2,3,4])
t = torch.tensor(a)
t = torch.topk(t,5)[1]
print(t)
one_hot = torch.nn.functional.one_hot(t, num_classes = 10)
print(one_hot)