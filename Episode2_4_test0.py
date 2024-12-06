import torch
import torch.nn as nn

in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

weight_matrix = torch.tensor([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6]
], dtype=torch.float32)

print(weight_matrix.matmul(in_features))

fc = nn.Linear(in_features=4, out_features=3, bias=False)
fc.weight = nn.Parameter(weight_matrix)
print(fc(in_features))
