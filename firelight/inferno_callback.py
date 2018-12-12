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
        self.log_interval_factor = 1.2
        self.min_log_interval = 20
        self.logged_last = {'train': None, 'val': None}

    @property
    def logger(self):
        assert self.trainer is not None
        assert hasattr(self.trainer, 'logger')
        return self.trainer.logger

    @property
    def log_now(self):
        phase = 'train' if self.trainer.model.training else 'val'
        i = self.trainer.iteration_count
        logged_last = self.logged_last[phase]
        if logged_last is None or (i >= self.log_interval_factor * logged_last
                      and i - logged_last >= self.min_log_interval):
            self.logged_last[phase] = i
            return True
        else:
            return False

    def get_trainer_states(self):
        states = ['inputs', 'error', 'target', 'prediction', 'loss']
        # TODO: better way to determine in what phase trainer is?
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
        # TODO: make Tensorboard logger accept rgb images
        #self.logger.log_object(self.name, image)

    def end_of_training_iteration(self, **_):
        # log_now = self.logger.log_images_now  # TODO: ask Nasim about this

        # log_now = self.logger.log_images_every.match(
        #    iteration_count=self.trainer.iteration_count,
        #    epoch_count=self.trainer.epoch_count,
        #    persistent=False)

        if self.log_now:
            self.do_logging()

    def end_of_validation_run(self, **_):
        if self.log_now:
            self.do_logging()


def get_visualization_callback(config):
    # a visualization config is parsed like this:
    # 1. input formats and global slicing is read
    # 2. the visualizers are processed:\
    #       - check if ContainerVisualizer -> if yes costruct sub-visualizers
    #       - else, just pass arguments and construct visualizer
    config = yaml2dict(config)
    visualizers = {}
    for name, args in config.items():
        visualizer = get_visualizer(args)
        visualizers[name] = visualizer
    callback = VisualizationCallback(visualizers)
    return callback
