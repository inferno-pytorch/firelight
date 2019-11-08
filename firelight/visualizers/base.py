import torch
from ..utils.dim_utils import SpecFunction, convert_dim, equalize_shapes
from .colorization import Colorize
from copy import copy
import torch.nn.functional as F
import logging
import sys
from pydoc import locate

# Set up logger
logging.basicConfig(format='[+][%(asctime)-15s][VISUALIZATION]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_single_key_value_pair(d):
    """
    Get the key and value of a length one dictionary.

    Parameters
    ----------
    d : dict
        Single element dictionary to split into key and value.

    Returns
    -------
    tuple
        of length 2, containing the key and value

    Examples
    --------

    >>> d = dict(key='value')
    >>> get_single_key_value_pair(d)
    ('key', 'value')

    """
    assert isinstance(d, dict), f'{d}'
    assert len(d) == 1, f'{d}'
    return list(d.items())[0]


def list_of_dicts_to_dict(list_of_dicts):
    """
    Convert a list of one element dictionaries to one dictionary.

    Parameters
    ----------
    list_of_dicts : :obj:`list` of :obj:`dict`
        List of one element dictionaries to merge.

    Returns
    -------
        dict

    Examples
    --------
    >>> list_of_dicts_to_dict([{'a': 1}, {'b': 2}])
    {'a': 1, 'b': 2}
    """

    result = dict()
    for d in list_of_dicts:
        key, value = get_single_key_value_pair(d)
        result[key] = value
    return result


def parse_slice(slice_string):
    """
    Parse a slice given as a string.

    Parameters
    ----------
    slice_string : str
        String describing the slice. Format as in fancy indexing: 'start:stop:end'.

    Returns
    -------
        slice

    Examples
    --------

    Everything supported in fancy indexing works here, too:

    >>> parse_slice('5')
    slice(5, 6, None)
    >>> parse_slice(':5')
    slice(None, 5, None)
    >>> parse_slice('5:')
    slice(5, None, None)
    >>> parse_slice('2:5')
    slice(2, 5, None)
    >>> parse_slice('2:5:3')
    slice(2, 5, 3)
    >>> parse_slice('::3')
    slice(None, None, 3)

    """
    # Remove whitespace
    slice_string.replace(' ', '')
    indices = slice_string.split(':')
    if len(indices) == 1:
        start, stop, step = indices[0], int(indices[0]) + 1, None
    elif len(indices) == 2:
        start, stop, step = indices[0], indices[1], None
    elif len(indices) == 3:
        start, stop, step = indices
    else:
        raise RuntimeError
    # Convert to ints
    start = int(start) if start != '' else None
    stop = int(stop) if stop != '' else None
    step = int(step) if step is not None and step != '' else None
    # Build slice
    return slice(start, stop, step)


def parse_named_slicing(slicing, spec):
    """
    Parse a slicing as a list of slice objects.

    Parameters
    ----------
    slicing : str or list or dict
        Specifies the slicing that is to be applied. Depending on the type:

        - :obj:`str`:  slice strings joined by ','. In this case, spec will be ignored. (e.g. :code:`'0, 1:4'`)
        - :obj:`list`: has to be list of one element dictionaries, that will be converted to one dict
          with :func:`list_of_dicts_to_dict`
        - :obj:`dict`: keys are dimension names, values corresponding slices (as strings)
          (e.g. :code:`{'B': '0', 'C': '1:4'}`)

    spec : list
        List of names of dimensions of the tensor that is to be sliced.

    Returns
    -------
    list
        List of slice objects

    Examples
    --------

        Three ways to encode the same slicing:

        >>> parse_named_slicing(':5, :, 1', ['A', 'B', 'C'])
        [slice(None, 5, None), slice(None, None, None), slice(1, 2, None)]
        >>> parse_named_slicing({'A': ':5', 'C': '1'}, ['A', 'B', 'C'])
        [slice(None, 5, None), slice(None, None, None), slice(1, 2, None)]
        >>> parse_named_slicing([{'A': ':5'}, {'C': '1'}], ['A', 'B', 'C'])
        [slice(None, 5, None), slice(None, None, None), slice(1, 2, None)]

    """
    if slicing is None:
        return slicing
    elif isinstance(slicing, str):  # No dimension names given, assume this is the whole slicing as one string
        # Remove whitespace
        slicing = slicing.replace(' ', '')
        # Parse slices
        slices = [parse_slice(s) for s in slicing.split(',')]
        assert len(slices) <= len(spec)
        return list(slices)
    elif isinstance(slicing, list):
        # if slicing is list, assume it is list of one element dictionaries (something like [B:0, C: '0:3'] in config)
        slicing = list_of_dicts_to_dict(slicing)

    assert isinstance(slicing, dict)
    # Build slice objects
    slices = []
    for d in spec:
        if d not in slicing:
            slices.append(slice(None, None, None))
        else:
            slices.append(parse_slice(str(slicing[d])))
    # Done.
    return slices


def parse_pre_func(pre_info):
    """
    Parse the pre-processing function for an input to a visualizer
    (as given by the 'pre' key in the input_mapping).

    Parameters
    ----------
    pre_info: list, dict or str
        Depending on the type:

        - :obj:`str`:  Name of function in torch, torch.nn.functional, or dotted path to function.
        - :obj:`list`: List of functions to be applied in succession. Each will be parsed by this function.
        - :obj:`dict`: Has to have length one. The key is the name of a function (see :obj:`str` above),
          the value specifies additional arguments supplied to that function (apart from the tensor that will be
          transformed). Either positional arguments can be specified as a list, or keyword arguments as a dictionary.

        Examples:

        - :code:`pre_info = 'sigmoid'`
        - :code:`pre_info = {'softmax': [1]}}`
        - :code:`pre_info = {'softmax': {dim: 0}}}`

    Returns
    -------
    Callable
        The parsed pre-processing function.

    """
    if isinstance(pre_info, list):
        # parse as concatenation
        funcs = [parse_pre_func(info) for info in pre_info]

        def pre_func(x):
            for f in funcs:
                x = f(x)
            return x

        return pre_func
    elif isinstance(pre_info, dict):
        pre_name, arg_info = get_single_key_value_pair(pre_info)
    elif isinstance(pre_info, str):
        pre_name = pre_info
        arg_info = []
    else:
        assert False, f'{pre_info}'

    if isinstance(arg_info, dict):
        kwargs = arg_info
        args = []
    elif isinstance(arg_info, list):
        kwargs = {}
        args = arg_info

    # Parse the function name
    pre_func_without_args = getattr(torch, pre_name, None)
    if pre_func_without_args is None:  # not found in torch
        pre_func_without_args = getattr(torch.nn.functional, pre_name, None)
    if pre_func_without_args is None:  # not found in torch or torch.nn.functional
        pre_func_without_args = locate(pre_name)
    assert callable(pre_func_without_args), f'Could not find the function {pre_name}'

    def pre_func(x):
        return pre_func_without_args(x, *args, **kwargs)

    return pre_func


# Default ways to label the dimensions depending on dimensionality  # TODO: make this easy to find
DEFAULT_SPECS = {
    3: list('BHW'),     # 3D: Batch, Height, Width
    4: list('BCHW'),    # 4D: Batch, Channel, Height, Width
    5: list('BCDHW'),   # 5D: Batch, Channel, Depth, Height, Width
    6: list('BCTDHW')   # 6D: Batch, Channel, Time, Depth, Height, Width
}
"""dict: The default ways to label the dimensions depending on dimensionality.

- 3 Axes : :math:`(B, H, W)`
- 4 Axes : :math:`(B, C, H, W)`
- 5 Axes : :math:`(B, C, D, H, W)`
- 6 Axes : :math:`(B, C, T, D, H, W)`

"""


def apply_slice_mapping(mapping, states, include_old_states=True):
    """
    Add/Replace tensors in the dictionary 'states' as specified with the dictionary 'mapping'. Each key in mapping
    corresponds to a state in the resulting dictionary, and each value describes:

    - from which tensors in `states` this state is grabbed (e.g. :code:`['prediction']`)
    - if a list of tensors is grabbed: which list index should be used (e.g :code:`'[index': 0]`)
    - what slice of the grabbed tensor should be used (e.g :code:`['B': '0', 'C': '0:3']`).
      For details see :func:`parse_named_slicing`.
    - what function in torch.nn.functional should be applied to the tensor after the slicing
      (e.g. :code:`['pre': 'sigmoid']`).
      See :func:`parse_pre_func` for details.

    These arguments can be specified in one dictionary or a list of length one dictionaries.

    Parameters
    ----------
    mapping: dict
        Dictionary describing the mapping of states
    states: dict
        Dictionary of states to be mapped. Values must be either tensors, or tuples of the form (tensor, spec).
    include_old_states: bool
        Whether or not to include states in the ouput dictionary, on which no operations were performed.

    Returns
    -------
    dict
        Dictionary of mapped states

    """
    mapping = copy(mapping)
    # assumes states are tuples of (tensor, spec) if included in mapping
    assert isinstance(states, dict)
    if include_old_states:
        result = copy(states)
    else:
        result = dict()
    if mapping is None:
        return result

    global_slice_info = mapping.pop('global', {})
    if isinstance(global_slice_info, list):
        global_slice_info = list_of_dicts_to_dict(global_slice_info)
        # add all non-scalar tensors to state mapping if global is specified
        for state_name in states:
            if state_name not in mapping:
                state = states[state_name]
                if isinstance(state, tuple):
                    state = state[0]
                if isinstance(state, list) and len(state) > 0:  # as e.g. inputs in inferno
                    state = state[0]
                if not isinstance(state, torch.Tensor):
                    continue
                if not len(state.shape) > 0:
                    continue
                mapping[state_name] = {}

    for map_to in mapping:
        map_from_info = mapping[map_to]

        # mapping specified not in terms of input tensor, but another visualizer
        if isinstance(map_from_info, BaseVisualizer):
            visualizer = map_from_info
            result[map_to] = visualizer(return_spec=True, **copy(states))
            continue

        if isinstance(map_from_info, str):
            map_from_key = map_from_info
            map_from_info = {}
        elif isinstance(map_from_info, (list, dict)):
            if isinstance(map_from_info, list) and isinstance(map_from_info[0], str):
                map_from_key = map_from_info[0]
                map_from_info = map_from_info[1:]
            else:
                map_from_key = map_to
            if isinstance(map_from_info, list):
                map_from_info = list_of_dicts_to_dict(map_from_info)

        # add the global slicing
        temp = copy(global_slice_info)
        temp.update(map_from_info)
        map_from_info = temp

        if map_from_key not in states:  # needed for container visualizers and 'visualization0'..
            continue

        # figure out state
        state_info = states[map_from_key]  # either (state, spec) or state
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        if not isinstance(state, (tuple, torch.Tensor)) and isinstance(state, list):
            index = map_from_info.pop('index', None)
            if index is not None:  # allow for index to be left unspecified
                index = int(index)
                state = state[index]
                assert isinstance(state, torch.Tensor), f'{map_from_key}, {type(state)}'
        if 'pre' in map_from_info:
            pre_func = parse_pre_func(map_from_info.pop('pre'))
        else:
            pre_func = None
        # figure out spec
        if 'spec' in map_from_info:
            spec = list(map_from_info.pop('spec'))
        else:
            if isinstance(state_info, tuple):
                spec = state_info[1]
            else:
                dimensionality = len(state.shape) if isinstance(state, torch.Tensor) else len(state[0].shape)
                assert dimensionality in DEFAULT_SPECS, f'{map_from_key}, {dimensionality}'
                spec = DEFAULT_SPECS[dimensionality]
        # get the slices
        map_from_slices = parse_named_slicing(map_from_info, spec)
        # finally map the state
        if isinstance(state, torch.Tensor):
            assert len(state.shape) == len(spec), f'{state.shape}, {spec} ({map_from_key})'
            state = state[map_from_slices].clone()
        elif isinstance(state, list):
            assert all(len(s.shape) == len(spec) for s in state), f'{[s.shape for s in state]}, {spec} ({map_from_key})'
            state = [s[map_from_slices] for s in state]
        else:
            assert False, f'state has to be list or tensor: {map_from_key}, {type(state)}'

        if pre_func is None:
            result[map_to] = (state, spec)
        else:
            result[map_to] = (pre_func(state), spec)
    return result


class BaseVisualizer(SpecFunction):
    """
    Base class for all visualizers.
    If you want to use outputs of other visualizers, derive from ContainerVisualizer instead.

    Parameters
    ----------
    input: list or None
        If the visualizer has one input only, this can be used to specify which state to pass (in the format of a
        value in input_mapping).
    input_mapping : dict or list
        Dictionary specifying slicing and renaming of states for visualization (see :func:`apply_slice_mapping`).
    colorize : bool
        If False, the addition/rescaling of a 'Color' dimension to RGBA in [0,1] is suppressed.
    cmap : str or callable
        If string, specifies the name of the matplotlib `colormap
        <https://matplotlib.org/examples/color/colormaps_reference.html>`_
        to be used for colorization.

        If callable, must be a mapping from a [Batch x Pixels] to [Batch x Pixels x Color] :class:`numpy.ndarray` used
        for colorization.
    background_label : int or float
        If specified, pixels with this value (after :meth:`visualize`) will be colored with :paramref:`background_color`.
    background_color : float or list
        Specifies the color for the background_label. Will be interpreted as grey-value if float, and RGB or RGBA if
        list of length 3 or 4 respectively.
    opacity : float
        Opacity of visualization, see colorization.py.
    colorize_jointly : list of str
        A list containing names of dimensions. Sets of data points separated only in these dimensions will be scaled
        equally at colorization (such that they lie in [0, 1]). Not used if 'value_range' is specified.

        Default: :code:`['W', 'H', 'D']` (standing for Width, Height, Depth)

        Examples:

        - :code:`color_jointly = ['W', 'H']` :      Scale each image separately
        - :code:`color_jointly = ['B', 'W', 'H']` : Scale images corresponding to different samples in the batch
          equally, such that their intensities are comparable

    value_range : List
        If specified, the automatic scaling for colorization is overridden. Has to have 2 elements.
        The interval [value_range[0], value_range[1]] will be mapped to [0, 1] by a linear transformation.

        Examples:

        - If your network has the sigmoid function as a final layer, the data does not need to be scaled
          further. Hence :code:`value_range = [0, 1]` should be specified.
        - If your network produces outputs normalized between -1 and 1, you could set :code:`value_range = [-1, 1]`.

    verbose : bool
        If true, information about the state dict will be printed during visualization.
    **super_kwargs:
        Arguments passed to the constructor of SpecFunction, above all the dimension names of inputs and output of
        visualize()
    """
    def __init__(self, input=None, input_mapping=None, colorize=True,
                 cmap=None, background_label=None, background_color=None, opacity=1.0, colorize_jointly=None,
                 value_range=None, verbose=False, scaling_options=None,
                 **super_kwargs):

        in_specs = super_kwargs.get('in_specs')
        super(BaseVisualizer, self).__init__(**super_kwargs)

        # always have the requested states in input mapping, to make sure their shape is inferred (from DEFAULT_SPECS)
        # if not specified.
        in_specs = {} if in_specs is None else in_specs
        self.input_mapping = {name: name for name in in_specs}
        # if 'input' is specified, map it to the first and only name in in_specs
        if input is not None:
            assert len(in_specs) == 1, \
                f"Cannot use 'input' in Visualizer with multiple in_specs. Please pass input mapping containing " \
                f"{list(in_specs.keys())} to {type(self).__name__}."
            name = get_single_key_value_pair(in_specs)[0]
            self.input_mapping[name] = input
        # finally set the input_mapping as specified in 'input_mapping'
        if input_mapping is not None:
            if input is not None:
                assert list(in_specs.keys())[0] not in input_mapping, \
                    f"State specified in both 'input' and 'input_mapping' Please choose one."
            self.input_mapping.update(input_mapping)
        self.colorize = colorize
        self.colorization_func = Colorize(cmap=cmap, background_color=background_color,
                                          background_label=background_label, opacity=opacity,
                                          value_range=value_range, colorize_jointly=colorize_jointly,
                                          scaling_options=scaling_options)
        self.verbose = verbose

    def __call__(self, return_spec=False, **states):
        """
        Visualizes the data specified in the state dictionary, following these steps:

        - Apply the input mapping (using :func:`apply_input_mapping`),
        - Reshape the states needed for visualization as specified by in_specs at initialization. Extra dimensions
          are 'put into' the batch dimension, missing dimensions are added (This is handled in the base class,
          :class:`firelight.utils.dim_utils.SpecFunction`)
        - Apply :meth:`visualize`,
        - Reshape the result, with manipulations applied on the input in reverse,
        - If not disabled by setting :code:`colorize=False`, colorize the result,
          leading to RGBA output with values in :math:`[0, 1]`.

        Parameters
        ----------
        return_spec: bool
            If true, a list containing the dimension names of the output is returned additionally
        states: dict
            Dictionary including the states to be visualized.

        Returns
        -------
        result: torch.Tensor or (torch.Tensor, list)
            Either only the resulting visualization, or a tuple of the visualization and the corresponding spec
            (depending on the value of :code:`return_spec`).

        """
        logger.info(f'Calling {self.__class__.__name__}.')
        
        if self.verbose:
            print()
            print(f'states passed to {type(self)}:')
            for name, state in states.items():
                print(name)
                if isinstance(state, tuple):
                    print(state[1])
                    if hasattr(state[0], 'shape'):
                        print(state[0].shape)
                    elif isinstance(state[0], list):
                        for s in state[0]:
                            print(s.shape)
                else:
                    print(type(state))

        # map input keywords and apply slicing
        states = apply_slice_mapping(self.input_mapping, states)

        if self.verbose:
            print()
            print(f'states after slice mapping:')
            for name, state in states.items():
                print(name)
                if isinstance(state, tuple):
                    print(state[1])
                    if hasattr(state[0], 'shape'):
                        print(state[0].shape)
                    elif isinstance(state[0], list):
                        for s in state[0]:
                            print(s.shape)
                else:
                    print(type(state))

        # apply visualize
        result, spec = super(BaseVisualizer, self).__call__(**states, return_spec=True)

        # color the result, if not suppressed
        result = result.float()
        if self.colorize:
            if self.verbose:
                print('colorizing now:', type(self))
                print('result before colorization:', result.shape)
            out_spec = spec if 'Color' in spec else spec + ['Color']
            result, spec = self.colorization_func(tensor=(result, spec), out_spec=out_spec, return_spec=True)
        if self.verbose:
            print('result:', result.shape)
        if return_spec:
            return result, spec
        else:
            return result

    def internal(self, *args, **kwargs):
        # essentially rename internal to visualize
        return self.visualize(*args, **kwargs)

    def visualize(self, **states):
        """
        Main visualization function that all subclasses have to implement.

        Parameters
        ----------
        states : dict
            Dictionary containing states used for visualization. The states in in_specs (specified at initialization)
            will have dimensionality and order of dimensions as specified there.

        Returns
        -------
            torch.Tensor
        """
        pass


class ContainerVisualizer(BaseVisualizer):
    """
    Base Class for visualizers combining the outputs of other visualizers.

    Parameters
    ----------
    visualizers : List of BaseVisualizer
        Child visualizers whose outputs are to be combined.
    in_spec : List of str
        List of dimension names. The outputs of all the child visualizers will be brought in this shape to be
        combined (in combine()).
    out_spec : List of str
        List of dimension names of the output of combine().
    extra_in_specs : dict
        Dictionary containing lists of dimension names for inputs of combine that are directly taken from the state
        dictionary and are not the output of a child visualizer.
    input_mapping : dict
        Dictionary specifying slicing and renaming of states for visualization (see :func:`apply_slice_mapping`).
    equalize_visualization_shapes : bool
        If true (as per default), the shapes of the outputs of child visualizers will be equalized by repeating
        along dimensions with shape mismatches. Only works if the maximum size of each dimension is divisible by the
        sizes of all the child visualizations in that dimension.
    colorize : bool
        If False, the addition/rescaling of a 'Color' dimension to RGBA in [0,1] is suppressed.
    **super_kwargs :
        Dictionary specifying other arguments of BaseVisualizer.

    """
    def __init__(self, visualizers, in_spec, out_spec, extra_in_specs=None, input_mapping=None,
                 equalize_visualization_shapes=True,
                 colorize=False, **super_kwargs):
        self.in_spec = in_spec
        self.visualizers = visualizers
        self.n_visualizers = len(visualizers)
        self.visualizer_kwarg_names = ['visualized_' + str(i) for i in range(self.n_visualizers)]
        if in_spec is None:
            in_specs = None
        else:
            in_specs = dict() if extra_in_specs is None else extra_in_specs
            in_specs.update({self.visualizer_kwarg_names[i]: in_spec for i in range(self.n_visualizers)})
        super(ContainerVisualizer, self).__init__(
            input_mapping={},
            in_specs=in_specs,
            out_spec=out_spec,
            colorize=colorize,
            **super_kwargs
        )
        self.container_input_mapping = input_mapping
        self.equalize_visualization_shapes = equalize_visualization_shapes

    def __call__(self, return_spec=False, **states):
        """
        Like call in BaseVisualizer, but computes visualizations for all child visualizers first, which will be passed
        to combine() (equivalent of visualize for BaseVisualizer).

        Parameters
        ----------
        return_spec: bool
            If true, a list containing the dimension names of the output is returned additionally
        states: dict
            Dictionary including the states to be visualized.

        Returns
        -------
            torch.Tensor or (torch.Tensor, list), depending on the value of :obj:`return_spec`.

        """
        states = copy(states)
        # map input keywords and apply slicing
        states = apply_slice_mapping(self.container_input_mapping, states)
        # apply visualizers and update state dict
        in_states = states.copy()
        visualizations = []
        for i in range(self.n_visualizers):
            visualizations.append(self.visualizers[i](**in_states, return_spec=True))
        if self.equalize_visualization_shapes:
            # add dimensions and reapeat them to make shapes of all visualizations match
            visualizations = equalize_shapes(tensor_spec_pairs=visualizations)
        for i, v in enumerate(visualizations):
            states[self.visualizer_kwarg_names[i]] = visualizations[i]
        return super(ContainerVisualizer, self).__call__(**states, return_spec=return_spec)

    def internal(self, **states):
        visualizations = []
        for name in self.visualizer_kwarg_names:
            visualizations.append(states[name])
        return self.combine(*visualizations, **states)

    def combine(self, *visualizations, **extra_states):
        """
        Main visualization function that all subclasses have to implement.

        Parameters
        ----------
        visualizations : :obj:`list` of :obj:`torch.Tensor`
            List containing the visualizations from the child visualizers. Their dimensionality and order of dimensions
            will be as specified in in_spec at initialization.
        extra_states : dict
            Dictionary containing extra states (not outputs of child visualizers) used for visualization. The states in
            :obj:`extra_in_specs` (specified at initialization) will have dimensionality and order of dimensions as
            specified there.

        Returns
        -------
            torch.Tensor

        """
        raise NotImplementedError
