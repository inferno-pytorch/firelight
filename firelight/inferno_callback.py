from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from .utils.io_utils import yaml2dict
from .config_parsing import get_visualizer
import torch
import logging
import sys

# Set up logger
logging.basicConfig(format='[+][%(asctime)-15s][VISUALIZATION]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _remove_alpha(tensor, background_brightness=1):
    return torch.ones_like(tensor[..., :3]) * background_brightness * (1-tensor[..., 3:4]) + \
           tensor[..., :3] * tensor[..., 3:4]


class VisualizationCallback(Callback):
    # Autodoc does not pick up VisualizationCallback, since Callback is mocked.

    VISUALIZATION_PHASES = ['training', 'validation']
    TRAINER_STATE_PREFIXES = ('training', 'validation')

    def __init__(self, logging_config, log_during='all'):
        super(VisualizationCallback, self).__init__()
        assert isinstance(logging_config, dict)
        self.logging_config = logging_config  # dictionary containing the visualizers as values with their names as keys

        # parse phases during which to log the individual visualizers
        for i, name in enumerate(logging_config):
            phases = logging_config[name].get('log_during', log_during)
            if isinstance(phases, str):
                if phases == 'all':
                    phases = self.VISUALIZATION_PHASES
                else:
                    phases = [phases]
            assert isinstance(phases, (list, tuple)), f'{phases}, {type(phases)}'
            assert all(phase in self.VISUALIZATION_PHASES for phase in phases), \
                f'Some phase not recognized: {phases}. Valid phases: {self.VISUALIZATION_PHASES}'
            logging_config[name]['log_during'] = phases

        # parameters specifying logging iterations
        # self.logged_last = {'train': None, 'val': None}

    def get_trainer_states(self):
        current_pre = self.TRAINER_STATE_PREFIXES[0 if self.trainer.model.training else 1]
        ignore_pre = self.TRAINER_STATE_PREFIXES[1 if self.trainer.model.training else 0]
        result = {}
        for key in self.trainer._state:
            if key.startswith(ignore_pre):
                continue
            state = self.trainer.get_state(key)
            if key.startswith(current_pre):
                key = '_'.join(key.split('_')[1:])  # remove current prefix
            if isinstance(state, torch.Tensor):
                state = state.cpu().detach().clone().float()  # logging is done on the cpu, all tensors are floats
            if isinstance(state, (tuple, list)) and all([isinstance(t, torch.Tensor) for t in state]):
                state = list(t.cpu().detach().clone().float() for t in state)

            result[key] = state
        return result

    def do_logging(self, phase, **_):
        assert isinstance(self.trainer.logger, TensorboardLogger)
        writer = self.trainer.logger.writer
        pre = 'training' if self.trainer.model.training else 'validation'
        for name, config in self.logging_config.items():
            if phase not in config['log_during']:  # skip visualizer if logging not requested for this phase
                continue
            visualizer = config['visualizer']
            logger.info(f'Logging now: {name}')
            image = _remove_alpha(visualizer(**self.get_trainer_states())).permute(2, 0, 1)  # to [Color, Height, Width]
            writer.add_image(tag=pre+'_'+name, img_tensor=image, global_step=self.trainer.iteration_count)
        logger.info(f'Logging finished')

    def end_of_training_iteration(self, **_):
        last_match_value = self.trainer.logger.log_images_every._last_match_value
        log_now = self.trainer.logger.log_images_every.match(
            iteration_count=self.trainer.iteration_count,
            epoch_count=self.trainer.epoch_count,
            persistent=False)
        self.trainer.logger.log_images_every._last_match_value = last_match_value
        if log_now:
            self.do_logging('training')

    def end_of_validation_run(self, **_):
        self.do_logging('validation')


def get_visualization_callback(config):
    """
    Gets an :mod:`inferno` callback for logging of firelight visualizations.

    Uses the :class:`inferno.trainers.basic.Trainer` state dictionary as input for the visualizers.

    The logging frequency is taken from the trainer's
    :class:`inferno.trainers.callbacks.logging.tensorboard.TensorboardLogger`.



    Parameters
    ----------
    config : str or dict
        If :obj:`str`, will be converted to :obj:`dict` using `pyyaml
        <https://pyyaml.org/wiki/PyYAMLDocumentation>`_.

        If :obj:`dict`, the keys are the tags under which the visualizations
        will be saved in Tensorboard, while the values are the configuration
        dictionaries to get the visualizers producing these visualizations,
        using :func:`firelight.config_parsing.get_visualizer`.

    Returns
    -------
    :class:`inferno.trainers.callbacks.base.Callback`

    Examples
    --------
    The structure of a configuration file could look like this:

    .. code:: yaml

        # visualize model predictions
        predictions:
            RowVisualizer:
                ...

        # visualize something else
        fancy_visualization:
            RowVisualizer:
                ...

    This configuration would produce images that are saved under the tags :code:`predictions` and
    :code:`fancy_visualization` in Tensorboard.

    """
    config = yaml2dict(config)
    logging_config = {}
    default_phases = config.pop('log_during', 'all')
    for name, kwargs in config.items():
        log_during = kwargs.pop('log_during', default_phases)
        visualizer = get_visualizer(kwargs)
        logging_config[name] = dict(visualizer=visualizer, log_during=log_during)
    callback = VisualizationCallback(logging_config)
    return callback
