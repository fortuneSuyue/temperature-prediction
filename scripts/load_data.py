import pandas as pd
import matplotlib.pyplot as plt
import os


def load_csv(path='../dataset/mpi_roof_2009a.csv', encoding='utf8', use_col=15):
    data_frame = pd.read_csv(path, usecols=list(range(use_col)), encoding=encoding)
    return data_frame


def load_all_data(root_dir='../dataset', encoding='utf8', use_col=15):
    files = next(os.walk(root_dir))[2]
    dfs = []
    for item in files:
        # print(item)
        tmp = load_csv(os.path.join(root_dir, item), encoding=encoding, use_col=use_col)
        print(item, len(tmp.iloc[:, 0]))
        dfs.append(tmp)
    return pd.concat(dfs)


def resample(data_frame, use_col=15, steps=6):
    xs = list(data_frame.iloc[:, 0])[::steps]
    ys = []
    for i in range(1, use_col):
        ys.append(list(data_frame.iloc[:, i])[::steps])
    return xs, ys


if __name__ == '__main__':
    df = load_all_data()
    print(df)
    print(df.keys().values)
    print(df.count(), '\n', df.describe())
    _, y = resample(df)
    plt.plot(y[1])
    plt.xticks(rotation=90)  # 横坐标每个值旋转90度
    plt.show()
    """
    ['Date Time' 'p (mbar)' 'T (degC)' 'Tpot (K)' 'Tdew (degC)' 'rh (%)'
 'VPmax (mbar)' 'VPact (mbar)' 'VPdef (mbar)' 'sh (g/kg)'
 'H2OC (mmol/mol)' 'rho (g/m**3)' 'wv (m/s)' 'max. wv (m/s)' 'wd (deg)']
    """