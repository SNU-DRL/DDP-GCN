from workalendar.asia import SouthKorea

import numpy as np
import pandas as pd


def z_score(x, mean, std):
    return (x - mean) / std


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


### revised

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def datasetIdx(seq_len=12, forecasting_horizon=12, data_dates=('2018-04-01 00:00:00', '2018-04-30 23:00:00', '5min'), seed=1220):

    np.random.seed(seed)

    time_window = seq_len + forecasting_horizon
    start_date, end_date, freq = data_dates
    assert ''.join([i for i in freq if not i.isdigit()]) == 'min', "frequency must be in minute."

    freq_min = ''.join([i for i in freq if i.isdigit()])
    time_window_ashour = time_window / (60 / int(freq_min))

    # weekday only
    data_dates = pd.date_range(start=start_date, end=end_date, freq=freq, closed='left')
    data_dates = data_dates[data_dates.weekday < 5]
    # delete holidays
    year = np.arange(int(start_date.split('-')[0]), int(end_date.split('-')[0])+1)
    holidays = [SouthKorea().holidays(y) for y in year][0]
    holidays = [holiday[0] for holiday in holidays]
    data_dates = [day for day in data_dates if day.date not in holidays]
    # delete monday morning (0h~time_window --> avoid to use weekend data)
    data_dates = [day for day in data_dates if not ((day.weekday == 0) & (day.hour < time_window_ashour))]

    #train:val:test = 0.7:0.1:0.2
    tidx_train = int(0.7 * len(data_dates))
    tidx_test = int(0.2 * len(data_dates))
    tidx_val = len(data_dates) - tidx_train - tidx_test
    train_time = data_dates[:tidx_train]
    val_time = data_dates[(tidx_train+time_window):(tidx_train+tidx_val)]
    test_time = data_dates[(tidx_train+tidx_val+time_window):]

    return (train_time, val_time, test_time)

# (train_time, val_time, test_time) = datasetIdx()
# print(train_time[0]) #2018-04-02 00:00:00
# print(train_time[-1]) #2018-04-20 16:00:00
# print(test_time[0]) #2018-04-24 20:25:00
# print(test_time[-1]) #2018-04-30 22:55:00
# np.save("/home/keun/test_time.npy", test_time)

def generateGraphData(forecasting_horizon, seq_len, targetTimes, targetLinks, speed):

    xtimes = np.array([str(t) for t in speed.index])
    x, y = [], []
    for n in range(len(targetTimes)):
        targetTime = targetTimes[n]
        tidx = np.where(xtimes == str(targetTime))[0]
        len_window = forecasting_horizon + seq_len

        x.append(speed.iloc[np.arange(tidx - len_window + 1, tidx + 1)][targetLinks])

    x = np.stack(x, axis=0)
    x = np.expand_dims(x, -1)
    return x


def data_gen_gap(data_type, forecasting_horizon=12, seq_len=12):

    speed = pd.read_hdf("/data2/keun/traffic-dataset/gap.h5")
    linkIds = np.load("/data2/keun/traffic-dataset/dataset-final/linkIdDict.npy").item()[data_type]

    (train_time, val_time, test_time) = datasetIdx()
    train_time = train_time[1440:]

    seq_train = generateGraphData(forecasting_horizon, seq_len, train_time, linkIds, speed)
    seq_val = generateGraphData(forecasting_horizon, seq_len, val_time, linkIds, speed)
    seq_test = generateGraphData(forecasting_horizon, seq_len, test_time, linkIds, speed)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def data_gen(data_type, forecasting_horizon=12, seq_len=12):

    speed = pd.read_hdf("/data2/keun/traffic-dataset/dataset-final/spd_5min.h5")
    linkIds = np.load("/data2/keun/traffic-dataset/dataset-final/linkIdDict.npy").item()[data_type]

    (train_time, val_time, test_time) = datasetIdx()

    # np.save("/data2/keun/traffic-dataset/dataset-final/datasplit.npy",
    #         [train_time, val_time, test_time])

    seq_train = generateGraphData(forecasting_horizon, seq_len, train_time, linkIds, speed)
    seq_val = generateGraphData(forecasting_horizon, seq_len, val_time, linkIds, speed)
    seq_test = generateGraphData(forecasting_horizon, seq_len, test_time, linkIds, speed)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset

# data_gen("Urban1", forecasting_horizon=12, seq_len=12)

def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
