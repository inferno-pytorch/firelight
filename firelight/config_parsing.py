from .visualizers.base import BaseVisualizer, ContainerVisualizer
from .utils.io_utils import yaml2dict
from pydoc import locate
import logging
import sys

# List of available visualizers (without container visualizers)
from .visualizers.visualizers import \
    IdentityVisualizer, \
    PcaVisualizer, \
    MaskedPcaVisualizer, \
    TsneVisualizer, \
    SegmentationVisualizer, \
    InputVisualizer, \
    TargetVisualizer, \
    PredictionVisualizer, \
    MSEVisualizer, \
    RGBVisualizer, \
    MaskVisualizer, \
    ThresholdVisualizer, \
    ImageVisualizer, \
    NormVisualizer, \
    DiagonalSplitVisualizer, \
    CrackedEdgeVisualizer

# List of available container visualizers (visualizers acting on outputs of child visualizers)
from .visualizers.container_visualizers import \
    ImageGridVisualizer, \
    RowVisualizer, \
    ColumnVisualizer, \
    OverlayVisualizer, \
    RiffleVisualizer, \
    StackVisualizer


# set up logging
logging.basicConfig(format='[+][%(asctime)-15s][VISUALIZATION]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
parsing_logger = logging.getLogger(__name__)


def get_single_key_value_pair(d):
    """
    Returns the key and value of a one element dictionary, checking that it actually has only one element
    Parameters
    ----------
    d : dict

    Returns
    -------
    tuple

    """
    assert isinstance(d, dict), f'{d}'
    assert len(d) == 1, f'{d}'
    return list(d.keys())[0], list(d.values())[0]


def get_visualizer_class(name):
    """
    Parses the class of a visualizer from a String. If the name is not found in globals(), tries to import it.

    Parameters
    ----------
    name : str
        Name of a visualization class imported above, or dotted path to one (e.g. your custom visualizer in a different
        library).

    Returns
    -------
        type or None

    """
    if name in globals():  # visualizer is imported above
        return globals().get(name)
    else:  # dotted path is given
        visualizer = locate(name)
        assert visualizer is not None, f'Could not find visualizer "{name}".'
        assert issubclass(visualizer, BaseVisualizer), f'"{visualizer}" is no visualizer'


def get_visualizer(config, indentation=0):
    """
    Parses a yaml configuration file to construct a visualizer.

    Parameters
    ----------
    config : str or dict or BaseVisualizer
        Either path to yaml configuration file or dictionary (as constructed by loading such a file).
        If already visualizer, it is just returned.
    indentation : int
        How far logging messages arising here should be indented.
    Returns
    -------
        BaseVisualizer

    """
    if isinstance(config, BaseVisualizer):  # nothing to do here
        return config
    # parse config to dict (does nothing if already dict)
    config = yaml2dict(config)
    # name (or dotted path) and kwargs of visualizer have to be specified as key and value of one element dictionary
    name, kwargs = get_single_key_value_pair(config)
    # get the visualizer class from its name
    visualizer = get_visualizer_class(name)
    parsing_logger.info(f'Parsing {"  "*indentation}{visualizer.__name__}')
    if issubclass(visualizer, ContainerVisualizer):  # container visualizer: parse sub-visualizers first
        child_visualizer_config = kwargs['visualizers']
        assert isinstance(child_visualizer_config, (list, dict)), \
            f'{child_visualizer_config}, {type(child_visualizer_config)}'
        if isinstance(child_visualizer_config, dict):  # if dict, convert do list
            child_visualizer_config = [{key: value} for key, value in child_visualizer_config.items()]
        child_visualizers = []
        for c in child_visualizer_config:
            v = get_visualizer(c, indentation + 1)
            assert isinstance(v, BaseVisualizer), f'Could not parse visualizer: {c}'
            child_visualizers.append(v)
        kwargs['visualizers'] = child_visualizers

    # TODO: add example with nested visualizers
    input_mapping = kwargs.get('input_mapping', {})
    for map_to, map_from in input_mapping.items():
        # visualizer has to be specified as one element dict
        if not (isinstance(map_from, dict) and len(map_from) == 1):
            continue
        name, _ = get_single_key_value_pair(map_from)
        # check if the name can be parsed as a visualizer
        try:
            _ = get_visualizer_class(name)
        except AssertionError:
            continue
        # parse the visualizer
        input_mapping[map_to] = get_visualizer(map_from, indentation+1)

    return visualizer(**kwargs)
