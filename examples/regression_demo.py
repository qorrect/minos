from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import boston_housing
import keras.metrics as metrics
import math

from minos.experiment.experiment import ExperimentSettings
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.model import Objective, Optimizer, Metric
from minos.model.parameter import int_param, float_param

from minos.train.utils import SimpleBatchIterator, CpuEnvironment
from minos.train.utils import GpuEnvironment
from minos.utils import load_best_model

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()


def search_model(experiment_label, steps, batch_size=1):
    y_train_log = [math.log(y) for y in y_train ]
    y_test_log = [math.log(y) for y in y_test ]
    batch_iterator = SimpleBatchIterator(X_train, y_train_log, batch_size=1, autoloop=True)
    test_batch_iterator = SimpleBatchIterator(X_test, y_test_log, batch_size=1, autoloop=True)
    from minos.experiment.experiment import Experiment

    from minos.model.model import Layout

    layout = Layout(
        X_train.shape[1],
        1,
        output_activation=None,
        block=[
            ('Dense'),
            ('Dropout'),
            ('Dense')
        ]
    )

    training = Training(
        Objective('mean_squared_error'),
        Optimizer(optimizer='RMSprop'),
        Metric('mean_squared_error'),
        EpochStoppingCondition(10),
        1)

    from minos.experiment.experiment import ExperimentParameters
    experiment_parameters = ExperimentParameters(use_default_values=True)
    experiment_settings = ExperimentSettings()

    experiment_parameters.layout_parameter('rows', 1)
    experiment_parameters.layout_parameter('blocks', 2)
    experiment_parameters.layout_parameter('layers', 1)
    experiment_parameters.layer_parameter('Dense.units', int_param(5, 50))
    experiment_parameters.layer_parameter('Dense.activation',None)
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
