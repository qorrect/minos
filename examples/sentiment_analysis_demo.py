from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

from minos.experiment.experiment import ExperimentSettings
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import *
from minos.model.model import Optimizer, Objective, Metric
from minos.model.parameter import int_param, float_param
from minos.train.utils import SimpleBatchIterator, GpuEnvironment
from minos.utils import load_best_model

batch_size = 32
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

model.fit(X_train, y_train, batch_size=batch_size, epochs=15, validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)


def search_model(experiment_label, steps, batch_size=1):
    batch_iterator = SimpleBatchIterator(X_train, y_train, batch_size=1, autoloop=True)
    test_batch_iterator = SimpleBatchIterator(X_test, y_test, batch_size=32, autoloop=True)
    from minos.experiment.experiment import Experiment

    from minos.model.model import Layout

    layout = Layout(
        X_train.shape[1],
        1,
        output_activation='sigmoid',
        block=[
            ('Embedding', {'input_dim': max_features, 'output_dim': 128}),
            ('LSTM', {'recurrent_dropout': 0.2, 'dropout': 0.2})

        ]
    )

    training = Training(
        Objective('binary_crossentropy'),
        Optimizer(optimizer='adam'),
        Metric('accuracy'),
        EpochStoppingCondition(10),
        1)

    from minos.experiment.experiment import ExperimentParameters
    experiment_parameters = ExperimentParameters(use_default_values=True)
    experiment_settings = ExperimentSettings()

    experiment_parameters.layout_parameter('rows', 1)
    experiment_parameters.layout_parameter('blocks', 1)
    experiment_parameters.layout_parameter('layers', 1)
    experiment_parameters.layer_parameter('Dense.units', int_param(5, 50))
    experiment_parameters.layer_parameter('Dense.activation', None)
    experiment_parameters.layer_parameter('Dropout.rate', float_param(0.1, 0.9))

    experiment_settings.ga['population_size'] = 25
    experiment_settings.ga['generations'] = 25
    # experiment_settings.ga['p_offspring'] = 0.75
    experiment_settings.ga['offsprings'] = 2
    experiment_settings.ga['p_mutation'] = .5

    experiment = Experiment(
        experiment_label,
        layout=layout,
        training=training,
        batch_iterator=batch_iterator,
        test_batch_iterator=test_batch_iterator,
        # environment=CpuEnvironment(n_jobs=12),
        environment=GpuEnvironment(['gpu:0', 'gpu:1'], n_jobs=12),
        parameters=experiment_parameters,
        settings=experiment_settings
    )

    run_ga_search_experiment(
        experiment,
        resume=False,
        log_level='DEBUG')
    load_best_model(experiment_label, steps - 1)


def main():
    label = 'regression_log_dropout_experiment'
    steps = 25

    search_model(label, steps)
    model = load_best_model(label, steps - 3)
    yhat = model.predict(X_test)
    for i in range(0, len(yhat)):
        yi = math.exp(yhat[i])
        print("Regression with Droupout = {}, Actual = {}".format(yi, math.exp(y_test[i])))


main()
