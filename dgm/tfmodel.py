import tensorflow as tf
import numpy as np

from .basemodel import BaseModel


class TFModel(BaseModel):
    def __init__(self, loader, columns):
        super().__init__(loader, columns)

        self.feature_columns = [tf.feature_column.numeric_column(k) for k in columns]

        print("Tensorflow Version {}".format(tf.__version__))
        tf.reset_default_graph()

    def build_model(self):
        layers = self.loader['layers']
        if len(layers) == 0 or layers[0] == 0:
            self.model = tf.estimator.LinearRegressor(feature_columns=self.feature_columns,
                                                      optimizer=tf.train.GradientDescentOptimizer(
                                                          learning_rate=self.loader['learning_rate']),
                                                      # optimizer=tf.train.ProximalAdagradOptimizer(
                                                      #     learning_rate=self.Loader['learning_rate'],
                                                      #     l2_regularization_strength=self.Loader['l1_regularization']
                                                      # ),
                                                      model_dir=self.loader.save_dir)
        else:
            self.model = tf.estimator.DNNRegressor(feature_columns=self.feature_columns,
                                                   hidden_units=layers,
                                                   optimizer=tf.train.ProximalAdagradOptimizer(
                                                       learning_rate=self.loader['learning_rate'],
                                                       l2_regularization_strength=self.loader['l1_regularization']
                                                   ),
                                                   dropout=self.loader['dropout'],
                                                   model_dir=self.loader.save_dir)
        return self.model

    def fit(self, feature_data, label_data):
        training_num = int(feature_data.shape[0]*self.loader['ration'])

        train_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=feature_data[:training_num],
            y=label_data[:training_num],
            batch_size=self.loader['batches'],
            epochs=self.loader['epochs'],
            shuffle=True)

        validate_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=feature_data[:training_num],
            y=label_data[:training_num],
            batch_size=self.loader['batches'],
            epochs=1,
            shuffle=True)

        if self.loader['train']:
            result = self.model.train(input_fn=train_input_fn)
            if self.loader['dump']:
                for var_name in self.model.get_variable_names():
                    print("{}: {}".format(var_name, self.model.get_variable_value(var_name)))

            accuracy_score = self.model.evaluate(input_fn=validate_input_fn)
            print("Training result {}".format(accuracy_score))
        else:
            result = None

        return result

    def validate(self, feature_data, label_data):
        validate_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=feature_data[:],
            y=label_data[:],
            epochs=1,
            shuffle=False)
        predicate_intput_fn = tf.estimator.inputs.pandas_input_fn(
            x=feature_data[:],
            epochs=1,
            shuffle=False)

        accuracy_score = self.model.evaluate(input_fn=validate_input_fn)
        print("Validation result {}".format(accuracy_score))

        predictions = self.model.evaluate(input_fn=predicate_intput_fn)
        mse , error, data = self.cost_fn(0, label_data[:], predictions)
        print("Validation result {}".format(accuracy_score))
        print("All MSE{}, Error {}".format(mse, error))

        np.savetxt(self.loader.save_dir + "/error.csv", np.asanyarray(data), delimiter=",")

    def predicate(self, feature_data, label_data):
        pass

    def test(self, feature_data, label_data):
        pass

    def summary(self):
        pass

    def cost_fn(self, offset, label_data, result):
        data=[]
        le, pe, cost = 0, 0, 0
        count = 0
        for i,p in enumerate(result):
            l_value = label_data[offset+i]
            p_value = p['predictions'][-1]
            cost += np.square(p_value - l_value)
            le += np.abs(l_value)
            pe += np.abs(p_value)
            count += 1
            data.append([p_value, l_value, (p_value - l_value)/l_value])
        cost /= count
        return cost, (pe-le)/le, data
