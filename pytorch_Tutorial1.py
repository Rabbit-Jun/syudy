import torch
import numpy as np


def p(x):
    print(f' {x}')

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# p(np_array)
# p(x_np)

x_ones = torch.ones_like(x_data)

x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Ones Tensor: \n {x_ones} \n")
# print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor.shape} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")