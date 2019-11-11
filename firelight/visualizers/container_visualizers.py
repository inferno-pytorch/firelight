from .base import ContainerVisualizer
from ..utils.dim_utils import convert_dim
import torch
import torch.nn.functional as F


def _to_rgba(color):
    """
    Converts a color to RGBA.

    Parameters
    ----------
    color : int, float or list
        If numeric, is interpreted as gray-value between 0 and 1. If list, has to have length 3 or 4 and is interpreted
        as RGB / RGBA depending on length (again, with values in [0, 1]).

    Returns
    -------
    list

    """
    if isinstance(color, (int, float)):  # color given as brightness
        result = [color, color, color, 1]
    elif isinstance(color, list):
        if len(color) == 3:  # color given as RGB
            result = color + [1]
        elif len(color) == 4:
            result = color.copy()
        else:
            assert False, f'len({color}) = {len(color)} has to be in [3, 4]'
    else:
        assert False, f'color specification not understood: {color}'
    return result


def _padded_concatenate(tensors, dim, pad_width, pad_value):
    """
    Concatenate tensors along specified dimension, adding padding between them.

    Parameters
    ----------
    tensors : list of torch.Tensor
        Tensors to be concatenated.
    dim : int
        Dimension along witch to concatenate.
    pad_width : int
        Width of the padding along concatenation dimension.
    pad_value : numeric or list like
        Value to fill the padding tensor with. Can be list, e.g. RGBA for tensors with color as last dimension.

    Returns
    -------
    torch.Tensor

    """
    tensors = list(tensors)
    device = tensors[0].device
    if pad_width != 0:
        pad_shape = list(tensors[0].shape)
        pad_shape[dim] = pad_width
        if isinstance(pad_value, list):
            pad_value = torch.Tensor(pad_value).to(device).type_as(tensors[0])
        pad_tensor = torch.ones(pad_shape).to(device) * pad_value
        [tensors.insert(i, pad_tensor) for i in range(len(tensors)-1, 0, -1)]
    return torch.cat(tensors, dim=dim)


class ImageGridVisualizer(ContainerVisualizer):
    """
    Visualizer that arranges outputs of child visualizers in a grid of images.

    Parameters
    ----------
    row_specs: list
        List of dimension names. These dimensions of the outputs of child visualizers will be put
        into the height dimension of the resulting image, according to the order in the list.

        In other words, data points only separated in dimensions at the beginning of this list will be right next to
        each other, while data points separated in dimensions towards the back will be further away from each other
        in the output image.

        A special dimension name is 'V' (for visualizers).
        It stands for the dimension differentiating between the child visualizers.

        **Example**:
        Given the tensor :code:`[[1,  2 , 3 ], [10, 20, 30]]` with shape (2, 3)
        and dimension names :code:`['A', 'B']`, this is the order of the rows, depending on the specified row_specs
        (suppose :code:`column_specs = []`):

        - If :code:`row_specs = ['B', 'A']`, the output will be :code:`[1, 2, 3, 10, 20, 30]`
        - If :code:`row_specs = ['A', 'B']`, the output will be :code:`[1, 10, 2, 20, 3, 30]`

    column_specs : list
        As row_specs but for columns of resulting image. Each dimension of child visualizations has to either
        occur in row_specs or column_specs. The intersection of row_specs and column specs has to be empty.
    pad_width : int or dict
        Determines the width of padding when concatenating images. Depending on type:

        - int:  Padding will have this width for concatenations along all dimensions, apart from H and W (no
                padding between adjacent pixels in image)
        - dict: Keys are dimension names, values the padding width when concatenating along them. Special key
                'rest' determines default value if given (otherwise no padding is used as default).

    pad_value : int or dict
        Determines the color of padding when concatenating images. Colors can be given as floats (gray values) or
        list of RGB / RGBA values. If dict, interpreted as pad_width
    upsampling_factor : int
        The whole resulting image grid will be upsampled by this factor. Useful when visualizing small images in
        tensorboard, but can lead to unnecessarily big file sizes.
    *super_args : list
    **super_kwargs : dict

    """
    def __init__(self, row_specs=('H', 'C', 'V'), column_specs=('W', 'D', 'T', 'B'),
                 pad_width=1, pad_value=.5, upsampling_factor=1, *super_args, **super_kwargs):
        super(ImageGridVisualizer, self).__init__(
            in_spec=None, out_spec=None,
            suppress_spec_adjustment=True,
            equalize_visualization_shapes=False,
            *super_args, **super_kwargs)
        assert all([d not in column_specs for d in row_specs]), 'every spec has to go either in rows or colums'

        # determine if the individual visualizers should be stacked as rows or columns
        if 'V' in row_specs:
            assert row_specs[-1] == 'V'
            row_specs = row_specs[:-1]
            self.visualizer_stacking = 'rows'
        elif 'V' in column_specs:
            assert column_specs[-1] == 'V'
            column_specs = column_specs[:-1]
            self.visualizer_stacking = 'columns'
        else:
            self.visualizer_stacking = 'rows'

        self.n_row_dims = len(row_specs)
        self.n_col_dims = len(column_specs)
        self.initial_spec = list(row_specs) + list(column_specs) + ['out_height', 'out_width', 'Color']

        self.pad_value = pad_value
        self.pad_width = pad_width

        self.upsampling_factor = upsampling_factor

    def get_pad_kwargs(self, spec):
        # helper function to manage padding widths and values
        result = dict()
        hw = ('H', 'W')
        if isinstance(self.pad_width, dict):
            result['pad_width'] = self.pad_width.get(spec, self.pad_width.get('rest', 0))
        else:
            result['pad_width'] = self.pad_width if spec not in hw else 0

        if isinstance(self.pad_value, dict):
            result['pad_value'] = self.pad_value.get(spec, self.pad_value.get('rest', .5))
        else:
            result['pad_value'] = self.pad_value if spec not in hw else 0
        result['pad_value'] = _to_rgba(result['pad_value'])

        return result

    def visualization_to_image(self, visualization, spec):
        # this function should not be overridden for regular container visualizers, but is here, as the specs have to be
        # known in the main visualization function. 'combine()' is never called, internal is used directly

        collapsing_rules = [(d, 'B') for d in spec if d not in self.initial_spec]  # everything unknown goes into batch
        visualization, spec = convert_dim(visualization, in_spec=spec, out_spec=self.initial_spec,
                                          collapsing_rules=collapsing_rules, return_spec=True)

        # collapse the rows in the 'out_width' dimension, it is at position -2
        for _ in range(self.n_row_dims):
            visualization = _padded_concatenate(visualization, dim=-3, **self.get_pad_kwargs(spec[0]))
            spec = spec[1:]

        # collapse the columns in the 'out_height' dimension, it is at position -3
        for _ in range(self.n_col_dims):
            visualization = _padded_concatenate(visualization, dim=-2, **self.get_pad_kwargs(spec[0]))
            spec = spec[1:]
        return visualization

    def internal(self, *args, return_spec=False, **states):
        images = []
        for name in self.visualizer_kwarg_names:
            images.append(self.visualization_to_image(*states[name]))

        if self.visualizer_stacking == 'rows':
            result = _padded_concatenate(images, dim=-3, **self.get_pad_kwargs('V'))
        else:
            result = _padded_concatenate(images, dim=-2, **self.get_pad_kwargs('V'))

        if self.upsampling_factor is not 1:
            result = F.interpolate(
                result.permute(2, 0, 1)[None],
                scale_factor=self.upsampling_factor,
                mode='nearest')
            result = result[0].permute(1, 2, 0)

        if return_spec:
            return result, ['H', 'W', 'Color']
        else:
            return result


class RowVisualizer(ImageGridVisualizer):
    """
    Visualizer that arranges outputs of child visualizers in a grid of images, with different child visualizations
    stacked vertically.
    For more options, see ImageGridVisualizer

    Parameters
    ----------
    *super_args :
    **super_kwargs :

    """
    def __init__(self, *super_args, **super_kwargs):
        super(RowVisualizer, self).__init__(
            row_specs=('H', 'S', 'C', 'V'),
            column_specs=('W', 'D', 'T', 'B'),
            *super_args, **super_kwargs)


class ColumnVisualizer(ImageGridVisualizer):
    """
    Visualizer that arranges outputs of child visualizers in a grid of images, with different child visualizations
    stacked horizontally (side by side).
    For more options, see ImageGridVisualizer

    Parameters
    ----------
    *super_args :
    **super_kwargs :

    """
    def __init__(self, *super_args, **super_kwargs):
        super(ColumnVisualizer, self).__init__(
            row_specs=('H', 'D', 'T', 'B'),
            column_specs=('W', 'S', 'C', 'V'),
            *super_args, **super_kwargs)


class OverlayVisualizer(ContainerVisualizer):
    """
    Visualizer that overlays the outputs of its child visualizers on top of each other, using transparency based on
    the alpha channel. The output of the first child visualizer will be on the top, the last on the bottom.

    Parameters
    ----------
    *super_args :
    **super_kwargs :

    """
    def __init__(self, *super_args, **super_kwargs):
        super(OverlayVisualizer, self).__init__(
            in_spec=['Color', 'B'],
            out_spec=['Color', 'B'],
            *super_args, **super_kwargs
        )

    def combine(self, *visualizations, **_):
        result = visualizations[-1]
        for overlay in reversed(visualizations[:-1]):
            a = (overlay[3] + result[3] * (1 - overlay[3]))[None]
            rgb = overlay[:3] * overlay[3][None] + result[:3] * result[3][None] * (1 - overlay[3][None])
            rgb /= a
            result = torch.cat([rgb, a], dim=0)
        return result


class RiffleVisualizer(ContainerVisualizer):
    """
    Riffles the outputs of its child visualizers along specified dimension.

    For a way to also scale target and prediction equally, have a look at StackVisualizer (if the range of
    values is known, you can also just use value_range: [a, b] for the child visualizers

    Parameters
    ----------
    riffle_dim : str
        Name of dimension which is to be riffled
    *super_args :
    **super_kwargs :

    Examples
    --------
    Riffle the channels of a multidimensional target and prediction, such that corresponding images are closer
    spatially. A possible configuration file would look like this::

        RiffleVisualizer:
            riffle_dim: 'C'
            visualizers:
                - ImageVisualizer:
                    input_mapping:
                        image: 'target'
                - ImageVisualizer:
                    input_mapping:
                        image: 'prediction'

    """
    def __init__(self, riffle_dim='C', *super_args, **super_kwargs):
        super(RiffleVisualizer, self).__init__(
            in_spec=[riffle_dim, 'B'],
            out_spec=[riffle_dim, 'B'],
            *super_args, **super_kwargs
        )

    def combine(self, *visualizations, **_):
        assert len(visualizations) > 0
        assert all(v.shape == visualizations[0].shape for v in visualizations[1:]), \
            f'Not all input visualizations have the same shape: {[v.shape for v in visualizations]}'
        result = torch.stack(visualizations, dim=1)
        result = result.contiguous().view(-1, visualizations[0].shape[1])
        return result


class StackVisualizer(ContainerVisualizer):
    """
    Stacks the outputs of its child visualizers along specified dimension.

    Parameters
    ----------
    stack_dim : str
        Name of new dimension along which the child visualizations will be stacked. None of the child visualizations
        should have this dimension.
    *super_args :
    **super_kwargs :

    Example
    -------
    Stack a multidimensional target and prediction along an extra dimension, e.g. 'TP'. In order to make target
    and prediction images comparable, disable colorization in the child visualizers and colorize only in the
    StackVisualizer, jointly coloring along 'TP', thus scaling target and prediction images by the same factors.
    The config would look like this::

        StackVisualizer:
            stack_dim: 'TP'
            colorize: True
            color_jointly: ['H', 'W', 'TP']  # plus other dimensions you want to scale equally, e.g. D = depth
            visualizers:
                - ImageVisualizer:
                    input_mapping:
                        image: 'target'
                    colorize = False
                - ImageVisualizer:
                    input_mapping:
                        image: 'target'
                    colorize = True

    """
    def __init__(self, stack_dim='S', *super_args, **super_kwargs):
        super(StackVisualizer, self).__init__(
            in_spec=[stack_dim, 'B'],
            out_spec=[stack_dim, 'B'],
            *super_args, **super_kwargs
        )

    def combine(self, *visualizations, **_):
        assert len(visualizations) > 0
        assert all(v.shape[1:] == visualizations[0].shape[1:] for v in visualizations[1:]), \
            f'Not all input visualizations have the same shape, apart from at dimension 0: ' \
            f'{[v.shape for v in visualizations]}'
        result = torch.cat(visualizations, dim=0)
        return result
