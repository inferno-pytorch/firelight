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
    def __init__(self, visualizers):
        super(VisualizationCallback, self).__init__()
        assert isinstance(visualizers, dict)
        self.visualizers = visualizers  # dictionary containing the visualizers as values with their names as keys

        # parameters specifying logging iterations
        self.logged_last = {'train': None, 'val': None}

    @property
    def logger(self):
        assert self.trainer is not None
        assert hasattr(self.trainer, 'logger')
        return self.trainer.logger

    def get_trainer_states(self):
        states = ['inputs', 'error', 'target', 'prediction', 'loss']
        pre = 'training' if self.trainer.model.training else 'validation'
        result = {}
        for s in states:
            state = self.trainer.get_state(pre + '_' + s)
            if isinstance(state, torch.Tensor):
                state = state.cpu().detach().clone().float()  # logging is done on the cpu, all tensors are floats
            result[s] = state
        return result

    def do_logging(self, **_):
        assert isinstance(self.logger, TensorboardLogger)
        writer = self.logger.writer
        pre = 'training' if self.trainer.model.training else 'validation'
        for name, visualizer in self.visualizers.items():
            logger.info(f'Logging now: {name}')
            image = _remove_alpha(visualizer(**self.get_trainer_states())).permute(2, 0, 1)  # to [Color, Height, Width]
            writer.add_image(tag=pre+'_'+name, img_tensor=image, global_step=self.trainer.iteration_count)
        logger.info(f'Logging finished')

    def end_of_training_iteration(self, **_):
        log_now = self.logger.log_images_every.match(
            iteration_count=self.trainer.iteration_count,
            epoch_count=self.trainer.epoch_count,
            persistent=False)
        if log_now:
            self.do_logging()

    def end_of_validation_run(self, **_):
        self.do_logging()


def get_visualization_callback(config):
    config = yaml2dict(config)
    visualizers = {}
    for name, args in config.items():
        visualizer = get_visualizer(args)
        visualizers[name] = visualizer
    callback = VisualizationCallback(visualizers)
    return callback
