
from keras.models import Model, clone_model, load_model
from keras.layers import Input, Dense, GRU, LSTM, LeakyReLU, Average, maximum, TimeDistributed, Subtract
from keras.callbacks import *
from keras.metrics import *
from keras.losses import *
from keras import optimizers
from keras import regularizers

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .basemodel import BaseModel

import numpy as np

import os, logging, copy


class KerasModel(BaseModel):
    def __init__(self, loader, columns):
        super().__init__(loader, columns)

    def __copy__(self):
        copied = type(self)(self.loader, self.columns)
        copied.model = clone_model(self.model)
        return copied

    def __deepcopy__(self, memo):
        copied = type(self)(copy.deepcopy(self.loader, memo),
                            copy.deepcopy(self.columns, memo))
        copied.model = clone_model(self.model)
        return copied

    def build_rnn_model(self):
        layers = self.loader['layers']

        auxiliary_input = Input(shape=(1,), name='aux_input')

        x = inputs = Input(shape=(self.loader.timestep, len(self.columns)))
        for layer in layers:
            if layer == 0:
                continue
            # x = Dense(layer, activation='sigmoid')(x)
            # x = LSTM(layer, return_sequences=True)(x)
            x = GRU(layer, return_sequences=False)(x)
        x = Dense(1, activation='relu')(x)
        # x = TimeDistributed(Dense(1, activation='relu'))(x)
        # x = Subtract()([x, auxiliary_input])
        outputs = x
        self.model = Model(inputs=[inputs], outputs=outputs)

        return self.model

    def build_dnn_model(self):
        layers = self.loader['layers']
        maxout = self.loader['maxout']

        x = inputs = Input(shape=len(self.columns))
        auxiliary_input = Dense(1, activation='relu', kernel_constraint='glorot_uniform')(x)
        for layer in layers:
            if layer == 0:
                continue
            # x = Dense(layer, activation='sigmoid')(x)
            if maxout < 2:
                x = Dense(layer, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
            else:
                x = maximum([Dense(layer,
                                   activation='sigmoid',
                                   kernel_regularizer=regularizers.l1(0.001),
                                   kernel_initializer='one'
                                   )(x) for _ in range(maxout)])
            x = LeakyReLU(alpha=0.1)(x)
        x = Dense(1, activation='relu')(x)
        x = Average()([x, auxiliary_input])
        outputs = LeakyReLU(alpha=-1)(x)

        self.model = Model(inputs=[inputs], outputs=outputs)

        return self.model

    def build_model(self):
        if os.path.exists(self.model_savename):
            self.model.load_model(self.model_savename)
            return self.model
        elif not self.loader['clean']:
            logging.warning("File not found %d, rebuild model", self.model_savename)

        optm = optimizers.Nadam()
        if self.loader['model'] == 'rnn':
            self.build_rnn_model()
        elif self.loader['model'] == 'dnn':
            self.build_dnn_model()
        else:
            return None

        self.model.compile(loss='mse', optimizers=optm, metrics=['mape'])

    def fit(self, feature_data, label_data):
        zero_data = np.zeros(len(label_data))
        training_num = int(feature_data.shape[0] * self.loader['ratio'])

        if self.loader['train']:
            save_best = ModelCheckpoint(self.model_pathname + "_.{epoch:02d}-{val_loss:.2f}.h5",
                                        verbose=1,
                                        save_best_only=True,
                                        period=10
                                        )
            csv_logger = CSVLogger(self.model_pathname + "_log.csv", append=True)
            es_logger = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=50,
                                      verbose=1,
                                      mode='auto')

            cb_logger = [csv_logger, save_best]
            if self.loader['earlystop']:
                cb_logger.append(es_logger)

            self.history = self.model.fit(feature_data[:training_num], label_data[:training_num],
                                          validation_data=(feature_data, label_data),
                                          batch_size=self.loader['batches'],
                                          epochs=self.loader['epochs'],
                                          verbose=1,
                                          callbacks=cb_logger)
            self.model.save(self.model_savename)

        return self.history

    def validate(self, feature_data, label_data):
        predictions = self.model.predict(feature_data, feature_data.shape[0], verbose=1)
        label_data = label_data.reshape((-1, 1))
        csv_data = np.concatenate((predictions, label_data, (predictions-label_data)/label_data), axis=1)
        np.savetxt(self.model_pathname + "_error.csv", np.asanyarray(csv_data), delimiter=",")

    def test(self, feature_data, label_data):
        seed = 7
        np.random.seed(seed)
        estimator = KerasRegressor(build_fn=self.build_model(),
                                   epochs=1,
                                   batch_size=self.loader['batches'],
                                   verbose=0
                                   )
        kfold = KFold(n_splits=5, random_state=seed)
        result = cross_val_score(estimator, feature_data, label_data, cv=kfold)
        print("Rsults: %.2f (%.2f) MSE" % (result.mean,(), result.std()))

    def summary(self):
        for column in self.columns:
            print(column)
        if self.model == "dnn" and self.loader["layers"][0] == 0:
            print(self.model.get_weights())
        print(self.model.summary())
