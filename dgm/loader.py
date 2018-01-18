import pandas as pd
import numpy as np
import os, shutil
import dgm

from .config import config
from .tfmodel import TFModel
from .kerasmodel import KerasModel


class Loader():
    def __init__(self, args):
        self.args = args
        self.parameters = config(args.f, args.cfg)
        self.api = self.parameters['api']
        self.model = self.parameters['model']
        self.timestep = self.parameters['timestep']
        self.cache = {}

    def __getitem__(self, item):
        return self.parameters[item]

    def load_feature(self, feature_file):
        feature_file = dgm.find_file(feature_file)
        with open(feature_file) as feature_desc:
            return False
        return False

    def load_csv(self, csv_file):
        csv_file = dgm.find_file(csv_file)
        with open(csv_file) as csv_data:
            data = pd.read_csv(csv_data)
            data.columns = data.columns.str.lower()
            self.cache['data'] = data
            print("Original data shape", data.shape)
            return True

        return False

    def load(self):
        self.save_dir = self.parameters['save_dir']
        self.layer_name = "_".join(str(x) for x in self.parameters['layers'])
        self.model_name = "{}_{}_{}_{}".format(self.api,
                                               self.model,
                                               "placeholder",
                                               self.layer_name)

        os.makedirs(self.save_dir, exist_ok=True)
        self.save_dir = self.save_dir + "/" + self.model_name
        if os.path.exists(self.save_dir):
            if self.parameters['clear']:
                shutil.rmtree(self.save_dir)
        else:
            os.makedirs(self.save_dir)

        self.parameters.save(self.save_dir)

        if not self.load_csv("csv_file"):
            return False

        return True

    def pd_filter(self):
        input_data = self.cache['data'].copy()
        valid_counters = input_data.columns
        label_column = "xxx" #TODO: set lable column

        if self.parameters['sort']:
            input_data.sort_values(by=label_column, ascending=False, inplace=True)
        input_data.to_csv(self.save_dir, + "input_data.csv", columns=valid_counters)

        if self.parameters['bias'] and self.model == 'dnn':
            input_data['bias'] = 1.0
            valid_counters.append('bias')

        label_data = input_data[label_column].reset_index(drop=True).values

        feature_data = input_data[valid_counters].reset_index(drop=True).astype(np.float32)

        if self.api == "keras":
            feature_data = feature_data.values

        if self.model == "rnn":
            # valid_counters.append(label_column)
            lookback_data = input_data[valid_counters]
            # feature_data, label_data = self.np_lookback(feature_data, label_data, self.timestep)
            feature_data = self.pd_lookback(lookback_data, valid_counters, self.timestep).values
            feature_data = feature_data.reshape((feature_data.shape[0], self.timestep, len(valid_counters)))

            label_data = label_data[self.timestep:]

        return feature_data, label_data, valid_counters

    def np_lookback(self, feature_data, label_data, timestep=1):
        label_data = label_data.reshape(-1,1)
        dataX, dataY = [], []
        for i in range(len(feature_data)-timestep-1):
            a = feature_data[i:(i+timestep), :]
            dataX.append(a)
            dataY.append(label_data[i+timestep,:])
        return np.array(dataX), np.array(dataY)

    def pd_lookback(self, df, counters, n_in=1, n_out=1, dropnan=True):
        n_vars = df.shape[1]
        n_obs =  n_in * n_vars
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [("{}_(t-{})".format(counters[j],i)) for j in range(n_vars)]
        for i in range(n_out):
            cols.append(df.shift(-i))
            if i==0:
                names += [("{}_(t)".format(counters[j])) for j in range(n_vars)]
            else:
                names += [("{}_(t+{})".format(counters[j],i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropnan(inplace=True)

        return agg.iloc[:, :n_obs] #, agg.iloc[:, -1]

    def build(self, columns):
        if self.api == "keras":
            modelobj = KerasModel(self, columns)
        elif self.api == "tf":
            modelobj = TFModel(self, columns)
        else:
            return None

        modelobj.build_model()

        return modelobj
