import time
import argparse
import torch

import Util
from Server import Server

def args_init_static():
    time_str = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))

    # 超参数
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=int, default=15, help="训练标签模式选择")
    parser.add_argument('--label_length', type=int, default=7, help="训练标签数量")
    parser.add_argument('--use_GPU', type=bool, default=False, help="是否使用GPU")
    parser.add_argument("--device_type",
                        type=torch.device,   default=None,    help="处理器类型")
    parser.add_argument('--num_clients', type=int, default=3, help="客户端数量")

    parser.add_argument('--LR', type=int, default=3, help="学习率")
    parser.add_argument('--batch_size', type=int, default=16, help="批处理尺寸")
    parser.add_argument('--episodes', type=int, default=10, help="数据集遍历次数")
    parser.add_argument('--tolerance', type=int, default=10, help="正确性判定区间")

    parser.add_argument('--net_dir', type=str,
                        default='./testmodel/',    help="模型保存/加载路径")
    parser.add_argument('--net_mark', type=str,
                        default=time_str,         help="模型名称标记")
    parser.add_argument('--data_dir', type=str,
                        default='C:\\Users\\bobby\\OneDrive\\桌面\\机器学习\\课程设计\\sleepdata.csv', help="数据集路径")
    args = parser.parse_args()
    '''
    check label_length
    label_list = ["Start year","Start day","End day","Time in bed","Wake up","Heart rate", "Activity (steps)"]
                   0            1           2         3             4         5             6
    '''
    args.label_length = Util.number_of_1(args.mode)

    return args


def use_GPU(args):
    # 定义是否使用GPU
    if args.use_GPU:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
            print("CUDA is available.")
        else:
            args.device = torch.device("cpu")
            print("CUDA is not available, fall back to CPU.")
    else:
        args.device = torch.device("cpu")


if __name__ == '__main__':
    args = args_init_static()
    mode_list = [127, 63, 111, 95, 31, 47, 15]
    for mode in mode_list:
        args.mode = mode
        args.label_length = Util.number_of_1(args.mode)
        time_str = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
        args.net_mark = time_str + "_" + str(mode)
        use_GPU(args)
        server = Server(args)
        server.start_train()
