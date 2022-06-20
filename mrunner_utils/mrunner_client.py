"""Parse and return mrunner gin-config and set-up Neptune.

This is copied from alpacka (with removed ray setup).
"""
import atexit
import datetime
import functools
import os
import pickle

import cloudpickle
import neptune


def get_configuration(spec_path):
    """Get mrunner experiment specification and gin-config overrides."""
    try:
        with open(spec_path, 'rb') as f:
            specification = cloudpickle.load(f)
    except pickle.UnpicklingError:
        with open(spec_path) as f:
            vars_ = {'script': os.path.basename(spec_path)}
            exec(f.read(), vars_)  # pylint: disable=exec-used
            specification = vars_['experiments_list'][0].to_dict()
            print('NOTE: Only the first experiment from the list will be run!')

    parameters = specification['parameters']
    gin_bindings = []
    for key, value in parameters.items():
        if isinstance(value, str) and not (value[0] == '@' or value[0] == '%'):
            binding = f'{key} = "{value}"'
        else:
            binding = f'{key} = {value}'
        gin_bindings.append(binding)

    return specification, gin_bindings


class NeptuneLogger:
    """Logs to Neptune."""

    def __init__(self, experiment):
        """Initialize NeptuneLogger with the Neptune experiment."""
        self._experiment = experiment

    def log_scalar(self, name, step, value):
        """Logs a scalar to Neptune."""
        self._experiment.send_metric(name, step, value)

    def log_image(self, name, step, img):
        """Logs an image to Neptune."""
        self._experiment.send_image(name, step, img)

    def log_property(self, name, value):
        """Logs a property to Neptune."""
        self._experiment.set_property(name, value)

    def log_text(self, name, content, show_on_screen):
        """Logs a text to Neptune."""
        self._experiment.log_text(name, content)


class NeptuneAPITokenException(Exception):
    def __init__(self):
        super().__init__('NEPTUNE_API_TOKEN environment variable is not set!')


def configure_neptune(specification):
    """Configures the Neptune experiment, then returns the Neptune logger."""
    if 'NEPTUNE_API_TOKEN' not in os.environ:
        raise NeptuneAPITokenException()

    git_info = specification.get('git_info', None)
    if git_info:
        git_info.commit_date = datetime.datetime.now()

    neptune.init(project_qualified_name=specification['project'])
    # Set pwd property with path to experiment.
    properties = {'pwd': os.environ.get("NEPTUNEPWD", os.getcwd())}
    neptune.create_experiment(name=specification['name'],
                              tags=specification['tags'],
                              params=specification['parameters'],
                              properties=properties,
                              git_info=git_info)
    atexit.register(neptune.stop)

    # This was uncommented in alpacka:
    # Add hook for Ray workers to make  them connect with appropriate neptune
    # experiment and set neptune logger.
    # def connect_to_neptune_experiment_add_logger(project_id, experiment_id):
    #     neptune.init(project_id)
    #     exp = neptune.project.get_experiments(
    #         id=experiment_id
    #     )[0]
    #     metric_logging.register_logger(NeptuneLogger(exp))
    #
    # ray.register_worker_init_hook(
    #     functools.partial(
    #         connect_to_neptune_experiment_add_logger,
    #         project_id=neptune.project.full_id,
    #         experiment_id=neptune.get_experiment().id,
    #     )
    # )

    return NeptuneLogger(neptune.get_experiment())
