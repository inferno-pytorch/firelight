from .visualizers.base import BaseVisualizer, ContainerVisualizer
from .utils.io_utils import yaml2dict
from pydoc import locate
import logging
import sys

from .visualizers.visualizers import \
    IdentityVisualizer, \
    PcaVisualizer, \
    MaskedPcaVisualizer, \
    SegmentationVisualizer, \
    InputVisualizer, \
    TargetVisualizer, \
    PredictionVisualizer, \
    MSEVisualizer, \
    RGBVisualizer, \
    MaskVisualizer, \
    ImageVisualizer, \
    NormVisualizer, \
    DiagonalSplitVisualizer, \
    CrackedEdgeVisualizer

from .visualizers.container_visualizers import \
    ImageGridVisualizer, \
    RowVisualizer, \
    ColumnVisualizer, \
    OverlayVisualizer, \
    RiffleVisualizer, \
    StackVisualizer


logging.basicConfig(format='[+][%(asctime)-15s][VISUALIZATION]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_single_key_value_pair(d):
    assert isinstance(d, dict), f'{d}'
    assert len(d) == 1, f'{d}'
    return list(d.keys())[0], list(d.values())[0]


def get_visualizer_class(name):
    assert name in globals(), f"Transform {name} not found."
    return globals().get(name)

def get_visualizer(config):
    config = yaml2dict(config)
    name, kwargs = get_single_key_value_pair(config)
    if name in globals():
        visualizer = get_visualizer_class(name)
    elif '.' in name:
        visualizer = locate(name)
        assert visualizer is not None, f'could not find {name}'
    else:
        return config
    logger.info(f'Parsing {visualizer.__name__}')
    if issubclass(visualizer, ContainerVisualizer):  # container visualizer: parse sub-visualizers first
        assert isinstance(kwargs['visualizers'], list), f'{kwargs["visualizers"]}, {type(kwargs["visualizers"])}'
        sub_visualizers = []
        for c in kwargs['visualizers']:
            v = get_visualizer(c)
            assert isinstance(v, BaseVisualizer), f'could not parse visualizer: {c}'
            sub_visualizers.append(v)
        kwargs['visualizers'] = sub_visualizers
    return visualizer(**kwargs)


if __name__ == '__main__':
    import pydoc

    def import_class(name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            logger.info(mod.__dict__)
            mod = getattr(mod, comp)
        return mod

    v = pydoc.locate('SegTags.visualizers.OrientationVisualizer')
    logger.info(v)

    assert False


    import torch
    import numpy as np

    config = './example_configs/test_config.yml'
    config = yaml2dict(config)
    callback = get_visualization_callback(config)
    v = callback.visualizer

    tensor = torch.Tensor(np.random.randn(2, 32, 10, 8, 8))
    result = v(inputs=tensor,
               prediction=2 * tensor,
               target=tensor > 0)
    print(result)
    print(result.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(result)
    plt.show()

