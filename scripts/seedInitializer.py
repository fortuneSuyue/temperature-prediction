import torch
import numpy as np
import random


# https://zhuanlan.zhihu.com/p/104019160
# https://blog.csdn.net/weixin_35097346/article/details/112018664
def randomSeedInitial(seed=256, cudnn_deterministic=True, cudnn_benchmark=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 为了保证可复现性, defaul True False; False True 可能可以提升gpu运行效率
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
