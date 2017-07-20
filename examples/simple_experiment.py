'''
Created on Feb 6, 2017

@author: julien
'''
from examples.ga.dataset import get_reuters_dataset
from minos.experiment.experiment import Experiment, ExperimentParameters, \
    load_experiment_blueprints, ExperimentSettings
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, AccuracyDecreaseStoppingCondition, \
    EpochStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.model import Objective, Optimizer, Metric, Layout
from minos.model.parameter import int_param, string_param, float_param
from minos.tf_utils import cpu_device, get_available_gpus
from minos.train.utils import CpuEnvironment, Environment, GpuEnvironment
import numpy as np

from minos.utils import load_best_model

np.random.seed(1337)
max_words = 1000





def search_model(experiment_label, steps, batch_size=32):
    """ This is where we put everythin together.
    We get the dataset, build the Training and Experiment objects, and run the experiment.
    The experiments logs are generated in ~/minos/experiment_label
    We use the CpuEnvironment to have the experiment run on the cpu, with 2 parralel processes.
    We could use GpuEnvironment to use GPUs, and specify which GPUs to use, and how many tasks
    per GPU
    """
    batch_iterator, test_batch_iterator, nb_classes = get_reuters_dataset(batch_size, max_words)
    layout = Layout(
        input_size=max_words,
        output_size=nb_classes,
        output_activation='softmax',
        block=[
            ('Dense', {'activation': 'relu'}),
            'Dropout',
            ('Dense')])
    training = Training(
        Objective('categorical_crossentropy'),
        Optimizer(optimizer='Adam'),
        Metric('categorical_accuracy'),
        EpochStoppingCondition(epoch=10),
        batch_size)
    settings = ExperimentSettings()
    # settings.search['layout'] = False
    settings.ga['population_size'] = 5
    settings.ga['generations'] = 5
    settings.ga['p_offspring'] = 0.75
    settings.ga['offsprings'] = 2
    settings.ga['p_mutation'] = 5
    settings.ga['mutation_std'] = 0.25


    """ Here we define the experiment parameters.
        We are using use_default_values=True, which will initialize
        all the parameters with their default values. These parameters are then fixed
        for the duration of the experiment and won't evolve.
        That means that we need to manually specify which parametres we want to test,
        and the possible values, either intervals or lists of values.
    
        If we want to test all the parameters and possible values, we can
        set use_default_values to False. In that case, random values will be generated
        and tested during the experiment. We can redefine some parameters if we want to
        fix their values.
        Reference parameters and default values are defined in minos.model.parameters
    
        We set the rows, blocks and layers parameters to 1 as we have specified a fixed layout.
        We also set the 'layout' search parameter to False to disable the layout search
        """
    parameters = ExperimentParameters(use_default_values=True)
    parameters.layout_parameter('blocks', 1)
    parameters.layout_parameter('rows', 2)
    parameters.layout_parameter('layers', 2)
    parameters.layer_parameter('Dense.units', int_param(10, 200))
    parameters.layer_parameter('Dense.activation', string_param(['relu']))
    parameters.layer_parameter('Dropout.rate', float_param(0.1, 0.9))

    experiment = Experiment(
        experiment_label,
        layout,
        training,
        batch_iterator,
        test_batch_iterator,
        GpuEnvironment(['gpu:0', 'gpu:1'], n_jobs=10),
        settings=settings,
        parameters=parameters)
    run_ga_search_experiment(
        experiment,
        resume=False,
        log_level='DEBUG')


def main():
    experiment_label = 'simple_retuers_classification_experiment'
    steps = 25
    search_model(experiment_label, steps)
    load_best_model(experiment_label, steps - 1)


if __name__ == '__main__':
    main()
