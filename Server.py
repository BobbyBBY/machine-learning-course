from Net import Net
import torch
import torch.nn as nn
import torch.optim as optim

from DataProcessor import DataProcessor
from Agent import Agent


class Server(object):
    def __init__(self, args):
        self.args = args
        self.num_clients = args.num_clients
        self.LR = args.LR
        self.episodes = args.episodes
        self.net_dir = args.net_dir
        self.net_mark = args.net_mark
        self.data_dir = args.data_dir
        self.label_length = args.label_length

        self.mode = args.mode
        self.tolerance = args.tolerance
        dataProcessor = DataProcessor(self.data_dir)
        self.trainloader, self.testloader = dataProcessor.read(self.mode)
        # 分割训练集
        self.dataset_list = list(self.trainloader)
        dataset_len = len(self.trainloader)
        self.epochs = dataset_len // self.num_clients
        # 创建全局模型
        self.global_net = Net(self.label_length)
        # 提取网络参数
        self.global_net_dict = self.global_net.state_dict()
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
        # 创建全局模型优化器
        self.global_optimizer = optim.SGD(
            self.global_net.parameters(), lr=self.LR)
        # 生成客户端
        self.agent_list = [
            Agent(self.LR, self.global_net_dict, self.label_length)]*self.num_clients
        # 创建客户端梯度收集列表
        self.client_grad_dict_list = [0] * self.num_clients

    def start_train(self):
        for episode in range(self.episodes):
            for epoch in range(self.epochs):
                self.global_net_dict = self.global_net.state_dict()  # 提取glodal网络参数

                # 通知agent开始训练，并接收梯度
                for index in range(self.num_clients):
                    agent = self.agent_list[index]
                    client_inputs, client_labels = self.dataset_list[epoch +
                                                                     index * self.epochs]
                    self.client_grad_dict_list[index] = agent.train(
                        client_inputs, client_labels, self.global_net_dict)

                # 聚合梯度
                # 取各client各层参数梯度均值
                client_average_grad_dict = dict()
                for key in self.global_net_dict:
                    for index in range(self.num_clients):
                        client_average_grad_dict[key] = self.client_grad_dict_list[index][key]*(
                            1/self.num_clients)

                # 更新global模型
                params_modules_server = self.global_net.named_parameters()
                for params_module in params_modules_server:
                    (name, params) = params_module
                    # 用字典中存储的子模型的梯度覆盖global中的参数梯度
                    params.grad = client_average_grad_dict[name]
                self.global_optimizer.step()

            # 每跑完一次epoch测试一下准确率
            # torch.no_grad()是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
            with torch.no_grad():
                total_correct = 0
                total_labels = 0
                total_loss = 0
                for data in self.testloader:
                    inputs, labels = data
                    labels = labels.view(len(labels)).long()
                    outputs = self.global_net(inputs)
                    total_loss += self.criterion(outputs, labels)
                    total_labels += labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    diff = (predicted - labels).abs()
                    total_correct += (diff < self.tolerance).int().sum().item()
                print('第%d个episode的识别准确率为：%d%% (误差区间：%d)' % (
                    episode + 1, (int)(100 * (float)(total_correct / total_labels)), self.tolerance))
                print('第%d个episode的平均loss为：%f' %
                      (episode + 1, (float)(total_loss / total_labels)))
        torch.save(self.global_net.state_dict(), '%s/net_%03d_%s.pth' %
                   (self.net_dir, self.episodes + 1, self.net_mark))
        print('successfully save the model to %s/net_%03d_%s.pth' %
              (self.net_dir, self.episodes + 1, self.net_mark))
