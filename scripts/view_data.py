import matplotlib.pyplot as plt
import pandas as pd


def read_show(path='../dataset/mpi_roof_2009a.csv', encoding='utf8', is_show=False):
    df = pd.read_csv(path, usecols=list(range(15)), encoding=encoding)
    print(path, '\n', df)
    col_names = df.columns.tolist()
    print(col_names)
    if is_show:
        for i in range(2, 3):
            plt.figure(dpi=300)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.plot(range(420451), df.iloc[:, i], 'm')
            plt.title(label=col_names[i])
            # plt.xticks(rotation=90)  # 横坐标每个值旋转90度
            plt.show()
        # di展示10天（1440=6*24*10
        plt.figure(dpi=300)
        plt.plot(df.iloc[: 6 * 24 * 10, 2], 'm')
        plt.show()
    return df


if __name__ == '__main__':
    read_show(path='../dataset/jena_climate_2009_2016.csv', is_show=True)
