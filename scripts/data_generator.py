import numpy as np
import torch
from torch.utils.data import Dataset


def get_mean_stds(ds: np.numarray, ax=0):
    return ds.mean(ax), ds.std(ax)


def data_regularization(ds: torch.FloatTensor, means: torch.FloatTensor, stds: torch.FloatTensor):
    """
    实参传递，直接改变numpy数组对象ds的值,也返回自身
    :param ds:
    :param means:
    :param stds:
    :return:
    """
    ds -= means
    ds /= stds
    return ds


class WeatherDataset(Dataset):
    def __init__(self, data: torch.FloatTensor, look_back=240, delay=24, min_index=0, max_index: int = None,
                 shuffle=False, use_col=14):
        """
        数据采样规则：
            训练集（打乱）：范围内随机选择采样起始点（有放回），x的长度为lookback（如10d），y（紧随x序列）的长度为delay（如1d）；\n
            验证集：依次采点，序列长度同训练集，采样点移动步长为batch_size，测预且仅预测一次未来的气温.
        :param data:
        :param look_back:
        :param delay: length of prediction
        :param min_index:
        :param max_index:
        :param shuffle: True for Training, False for Validation.
        """
        super(WeatherDataset, self).__init__()
        assert min_index >= 0
        assert len(data.shape) == 2
        assert data.shape[1] >= use_col
        self.min_index = min_index
        if max_index is None:
            self.max_index = data.shape[0]
        else:
            self.max_index = max_index
        assert self.max_index <= data.shape[0]
        self.look_back = look_back
        self.delay = delay
        self.use_col = use_col
        if shuffle:
            self.data_x, self.data_y = None, None
            self.data = data
        else:
            self.data = None
            self.data_x, self.data_y = (
                torch.zeros(size=(self.__len__(), self.look_back, self.use_col), dtype=torch.float32),
                torch.zeros(size=(self.__len__(), self.delay, 1), dtype=torch.float32))
            self._construction(data, shuffle)
        assert self.__len__() > 0

    def _construction(self, data: np.numarray, shuffle=False):
        assert not shuffle
        for i in range(self.__len__()):
            end_x = min(self.max_index-self.delay, self.min_index + self.delay*i + self.look_back)
            self.data_x[i, :, :] = data[end_x-self.look_back: end_x, : self.use_col]
            self.data_y[i, :, :] = data[end_x: end_x+self.delay, 1: 2]

    def __len__(self):
        length_seq = self.max_index - self.min_index - self.look_back
        assert length_seq >= self.delay
        if self.data is None:
            return (length_seq + self.delay - 1) // self.delay
        return length_seq - self.delay + 1

    def __getitem__(self, item):
        if self.data is not None:  # training_set
            start = np.random.randint(self.min_index, self.max_index-self.look_back-self.delay+1)
            data_x = self.data[start: start+self.look_back, :]
            start += self.look_back
            data_y = self.data[start: start+self.delay, 1: 2]
            return data_x, data_y
        else:
            return self.data_x[item], self.data_y[item]


if __name__ == '__main__':
    from load_data import load_all_data, resample
    n_col = 15  # 包含name
    _, a = resample(load_all_data(use_col=n_col), use_col=n_col)
    a = torch.FloatTensor(a).t()
    print(a.shape)
    train_size = a.shape[0] // 2
    val_size = train_size // 2
    test_size = a.shape[0] - train_size - val_size
    print(train_size, val_size, test_size)
    m, s = get_mean_stds(ds=a[: train_size])
    data_regularization(a, m, s)
    train_dataset = WeatherDataset(data=a, max_index=train_size, shuffle=True)
    val_dataset = WeatherDataset(data=a, min_index=train_size, max_index=train_size+val_size, shuffle=False)
    test_dataset = WeatherDataset(data=a, min_index=train_size+val_size, shuffle=False)
    sample = test_dataset.__getitem__(-1)[-1]
    print(sample, sample.shape)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
