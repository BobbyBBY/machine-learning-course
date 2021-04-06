import pandas as pd
import datetime
import numpy as np
from torch.utils.data import DataLoader
from sklearn import model_selection

import Util
from MyDataset import MyDataset


test_size = 0.2
batch_size = 32


class DataProcessor(object):
    def __init__(self, data_dir):
        # 使用open以识别中文路径
        self.df = pd.read_csv(open(data_dir), delimiter=";")
        # 查看空数据比例
        # self.df.info()
        self.label_list = ["Start year", "Start day", "End day",
                           "Time in bed", "Wake up", "Heart rate", "Activity (steps)"]

    def read(self, mode, output_length):
        # 时间string to datetime
        # 上床时间
        self.df["Start"] = pd.to_datetime(self.df["Start"])
        # 起床时间
        self.df["End"] = pd.to_datetime(self.df["End"])
        # 在床上的时间
        self.df["Time in bed"] = self.df["End"] - self.df["Start"]
        # 数据集说明了收集起始时间，实则使用默认的1970年也没问题，最后都要取模
        start_time_str = "2014-01-01"
        start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d")
        self.df["Start"] = self.df["Start"]-start_time
        self.df["End"] = self.df["End"]-start_time
        # 时间datetime to 年、日百分比
        # 比如2015-01-31-09:22:30,年百分比为31/266,日百分比为(((9*60+22)*60)+30)/86400
        self.df["Start year"] = (self.df["Start"].astype(
            "timedelta64[D]").astype(float) % 366)/366
        self.df["Start day"] = (self.df["Start"].astype(
            "timedelta64[s]").astype(float) % 86400)/86400
        self.df["End day"] = (self.df["End"].astype(
            "timedelta64[s]").astype(float) % 86400)/86400
        self.df["Time in bed"] = (self.df["Time in bed"].astype(
            "timedelta64[s]").astype(float) % 86400)/86400
        # 睡眠质量 string to int
        self.df["Sleep quality"] = self.df["Sleep quality"].apply(
            lambda x: np.nan if x in ["-"] else round((int(x[:-1]) * output_length)/100)).astype(int)

        # 由于存在许多空数据，根据MODE选择不同的label进行训练
        mode_bin = Util.int_to_str(mode)
        selected_list = []
        for i in range(len(mode_bin)):
            if mode_bin[i] == "1":
                selected_list.append(self.label_list[i])
        selected_list_y = selected_list.copy()
        selected_list_y.append("Sleep quality")
        df2 = self.df[selected_list_y]
        # 删除空数据
        df2 = df2.dropna()

        # 以下标签存在空数据，需要drop后再处理
        if "Wake up" in selected_list:
            # 起床心情
            """
            ) - you feeling great after night (good)
            :| - not so good (average)
            :( - terrible night (bad)
            """
            df2["Wake up"] = df2["Wake up"].replace(
                {":)": 2, ":|": 1, ":(": 0}).astype("int")
        if "Heart rate" in selected_list:
            # 睡眠平均心率
            df2["Heart rate"] = df2["Heart rate"].astype("int")
        if "Activity (steps)" in selected_list:
            # 睡前当日步数
            df2["Activity (steps)"] = df2["Activity (steps)"].astype("int")

        X = df2[selected_list].values
        # y的维度要于X匹配
        y = df2[["Sleep quality"]].values
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=test_size)
        # 定义训练数据集,构造Dataset对象
        trainset = MyDataset(X_train, y_train)
        # 定义训练批处理数据
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
        )
        # 定义测试数据集
        testset = MyDataset(X_test, y_test)
        # 定义测试批处理数据
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
        )
        return trainloader, testloader
