import torch

t1 = torch.tensor([1,1,1,1])
t2 = torch.tensor([2,2,2,2])
t3 = torch.tensor([3,3,3,3])

print(torch.cat((t1,t2,t3),dim=0))
# print(torch.cat((t1,t2,t3),dim=1))
# print(torch.cat((t1,t2,t3),dim=2))

print(torch.stack((t1,t2,t3),dim=0))
print(torch.stack((t1,t2,t3),dim=1))
# print(torch.stack((t1,t2,t3),dim=2))