import time
import torch 
from torch import nn

#---------------------------------------------------------------------------------------------------------
#------------------------------------------ 简单示例测试 --------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# 使用cpu
# a = torch.rand((10000,200))
# b = torch.rand((200,10000))
# tic = time.time()
# c = torch.matmul(a,b)
# toc = time.time()

# print(toc-tic)
# print(a.device)
# print(b.device)
# 输出
# 0.2060565948486328
# cpu
# cpu


# 使用gpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# a = torch.rand((10000,200),device = device) #可以指定在GPU上创建张量
# b = torch.rand((200,10000)) #也可以在CPU上创建张量后移动到GPU上
# b = b.to(device) #或者 b = b.cuda() if torch.cuda.is_available() else b 
# tic = time.time()
# c = torch.matmul(a,b)
# toc = time.time()
# print(toc-tic)
# print(a.device)
# print(b.device)
# 输出
# 0.07401633262634277
# cuda:0
# cuda:0

#---------------------------------------------------------------------------------------------------------



# 1，查看gpu信息
# 检查CUDA是否可用
if_cuda = torch.cuda.is_available()
print("if_cuda=", if_cuda)

# # 如果CUDA可用，获取GPU数量
# if if_cuda:
#     gpu_count = torch.cuda.device_count() # 如果CUDA可用，这个函数返回系统中可用的GPU数量。
#     print("gpu_count=", gpu_count)

#     # 获取并打印每个GPU的型号
#     for i in range(gpu_count):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 2，将张量在gpu和cpu间移动
tensor = torch.rand((100,100))
tensor_gpu = tensor.to("cuda:0") # 将张量移动到指定设备上
# 或者 tensor_gpu = tensor.cuda()  # 这是一个简写方式，默认将张量移动到第一个可用的GPU。

print(tensor_gpu.device)
print(tensor_gpu.is_cuda)

tensor_cpu = tensor_gpu.to("cpu") # 或者 tensor_cpu = tensor_gpu.cpu() 
print(tensor_cpu.device)
# 输出
# cuda:0
# True
# cpu

# 3，将模型中的全部张量移动到gpu上
net = nn.Linear(2,1)
print(next(net.parameters()).is_cuda) # 这行代码检查模型中的第一个参数张量是否在CUDA设备上。
net.to("cuda:0") # 将模型中的全部参数张量依次到GPU上，注意，无需重新赋值为 net = net.to("cuda:0")
# 这个方法将模型（包括其所有参数）移动到指定的GPU上。注意，这种方法会改变模型内部的张量，但不需要重新赋值。

print(next(net.parameters()).is_cuda) # 这个类用于封装模型，以支持多GPU数据并行。它会自动将模型复制到所有可用的GPU上。
print(next(net.parameters()).device) # 这是一个属性，返回参与数据并行的GPU的ID列表。
# 输出
# False
# True
# cuda:0

print("----------------------------------------------------------")
# 4，创建支持多个gpu数据并行的模型
linear = nn.Linear(2,1)
print(next(linear.parameters()).device)

model = nn.DataParallel(linear)
print(model.device_ids)
print(next(model.module.parameters()).device) 
# 当模型被nn.DataParallel封装后，原始模型可以通过model.module访问。这行代码打印原始模型中第一个参数张量所在的设备。

#注意保存参数时要指定保存model.module的参数
torch.save(model.module.state_dict(), "./data/model_parameter.pkl") 
# 这个函数保存模型的参数。由于模型被封装在nn.DataParallel中，我们使用model.module.state_dict()来获取原始模型的参数。

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load("./data/model_parameter.pkl")) 
print("----------------------------------------------------------")

# 5，清空cuda缓存

# 该方法在cuda超内存时十分有用
torch.cuda.empty_cache()
