import copy
import torch.nn as nn
import torch.optim as optim

from Net import Net

class Agent(object):
    def __init__(self, LR, global_net_dict, label_length, output_length):
        self.LR = LR
        self.output_length = output_length
        self.net = Net(label_length, self.output_length+1) # 由于取头取尾,所以加1
        self.net.load_state_dict(global_net_dict)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, inputs, labels, global_net_dict):
        labels = labels.view(len(labels)).long()
        # inputs, labels = inputs.to(device), labels.to(device)
        self.net.load_state_dict(global_net_dict)
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        optimizer = optim.SGD(self.net.parameters(), lr=self.LR)
        optimizer.zero_grad()  # 梯度置零
        loss.backward()  # 求取梯度
        # 提取梯度
        grad_dict = dict()
        params_modules = list(self.net.named_parameters())
        for params_module in params_modules:
            (name, params) = params_module
            params_grad = copy.deepcopy(params.grad)
            grad_dict[name] = params_grad
        optimizer.zero_grad()  # 梯度置零
        return grad_dict
