# LEnet 网络
################################
# linear  input
# linear  16
# linear  64
# linear  100
###################################
import torch.nn as nn

# 定义网络结构
class Net(nn.Module):
    def __init__(self, input):
        super(Net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input, 64),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU()
        )
        self.l3 = nn.Linear(512, 101)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
