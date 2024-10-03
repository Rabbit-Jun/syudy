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

# print(f"Random Tensor: \n {rand_tensor.shape} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")


tensor = torch.rand(3,4)

# p(tensor)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"dtype of tensor: {tensor.dtype}")
# print(f"device of tensor: {tensor.device}")

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    

tensor = torch.tensor([[1,2,3], [4,5,6],[7,8,9]])
# p(tensor)
# print('First row: ', tensor[0])
# print('First column: ', tensor[:, 0])
# print('Last column: ', tensor[..., -1])
tensor[: ,1]= 0
# print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)


y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.randint_like(tensor, low=0, high= 10)
torch.matmul(tensor, tensor.T, out=y3)
# print(f'y1: {y1}')
# print(f'y2: {y2}')
# print(f'y3: {y3}')

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.randint_like(tensor, low=0, high=10)
torch.mul(tensor, tensor, out= z3)
# print(f'z1: {z1}')
# print(f'z2: {z2}')
# print(f'z3: {z3}')


agg = tensor.sum()
agg_item = agg.item()

# print(agg, type(agg))
# print(agg_item, type(agg_item))

print(tensor, '\n')
tensor.add_(5)
print(tensor)