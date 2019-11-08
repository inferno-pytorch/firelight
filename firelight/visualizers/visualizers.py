from .base import BaseVisualizer
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn.functional import pad
try:
    import umap
    umap_available = True
except ImportError:
    umap_available = False


class IdentityVisualizer(BaseVisualizer):
    """
    Visualizer that returns the tensor passed to it. Useful to visualize each channel of a tensor as a separate image.
    """
    def __init__(self, **super_kwargs):
        super(IdentityVisualizer, self).__init__(
            in_specs={'tensor': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, tensor, **_):
        """"""
        return tensor


class ImageVisualizer(BaseVisualizer):
    """
    Same as :class:`IdentityVisualizer`, but acting on 'image'.
    """
    def __init__(self, **super_kwargs):
        super(ImageVisualizer, self).__init__(
            in_specs={'image': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, image, **_):
        """"""
        return image


class InputVisualizer(BaseVisualizer):
    """
    Same as :class:`IdentityVisualizer`, but acting on 'input'.
    """
    def __init__(self, **super_kwargs):
        super(InputVisualizer, self).__init__(
            in_specs={'input': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, input, **_):
        """"""
        return input


class TargetVisualizer(BaseVisualizer):
    """
    Same as :class:`IdentityVisualizer`, but acting on 'target'.
    """
    def __init__(self, **super_kwargs):
        super(TargetVisualizer, self).__init__(
            in_specs={'target': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, target, **_):
        """"""
        return target


class PredictionVisualizer(BaseVisualizer):
    """
    Same as :class:`IdentityVisualizer`, but acting on 'prediction'.
    """
    def __init__(self, **super_kwargs):
        super(PredictionVisualizer, self).__init__(
            in_specs={'prediction': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, prediction, **_):
        """"""
        return prediction


class MSEVisualizer(BaseVisualizer):
    """
    Visualize the Mean Squared Error (MSE) between two tensors (e.g. prediction and target).
    """
    def __init__(self, **super_kwargs):
        super(MSEVisualizer, self).__init__(
            in_specs={'prediction': 'B', 'target': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, prediction, target, **_):
        """"""
        return (prediction - target)**2


class SegmentationVisualizer(BaseVisualizer):
    """
    Same as :class:`IdentityVisualizer`, but acting on 'segmentation'.
    """
    def __init__(self, **super_kwargs):
        super(SegmentationVisualizer, self).__init__(
            in_specs={'segmentation': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, segmentation, **_):
        """"""
        return segmentation


class RGBVisualizer(BaseVisualizer):
    """
    Visualize the input tensor as RGB images. If the input has n * 3 channels, n color images will be returned.
    """
    def __init__(self, **super_kwargs):
        super(RGBVisualizer, self).__init__(
            in_specs={'tensor': ['B', 'C']},
            out_spec=['B', 'C', 'Color'],
            **super_kwargs
        )

    def visualize(self, tensor, **_):
        """"""
        n_channels = tensor.shape[1]
        assert n_channels % 3 == 0, f'the number of channels {tensor.shape[1]} has to be divisible by 3'
        tensor = tensor.contiguous().view(tensor.shape[0], n_channels // 3, 3)
        return tensor


class MaskVisualizer(BaseVisualizer):
    """
    Returns a mask that is 1 where the input image equals the mask label passed at initialization, and 0 elsewhere

    Parameters
    ----------
    mask_label : float
        Label to be used for the construction of the mask
    **super_kwargs

    """
    def __init__(self, mask_label, **super_kwargs):
        super(MaskVisualizer, self).__init__(
            in_specs={'tensor': ['B']},
            out_spec=['B'],
            **super_kwargs
        )
        self.mask_label = mask_label

    def visualize(self, tensor, **states):
        """"""
        return (tensor == self.mask_label).float()


class ThresholdVisualizer(BaseVisualizer):
    """
    Returns a mask resulting from thresholding the input tensor.

    Parameters
    ----------
    threshold : int or float
    mode : str, optional
        one of the :attr:`ThresholdVisualizer.MODES`, specifying how to threshold.
    super_kwargs
    """

    MODES = ['greater', 'smaller', 'greater_equal', 'smaller_equal']

    def __init__(self, threshold, mode='greater_equal', **super_kwargs):
        super(ThresholdVisualizer, self).__init__(
            in_specs={'tensor': ['B']},
            out_spec=['B'],
            **super_kwargs
        )
        self.threshold = threshold
        assert mode in ThresholdVisualizer.MODES, f'Mode {mode} not supported. Use one of {ThresholdVisualizer.MODES}'
        self.mode = mode

    def visualize(self, tensor, **_):
        """"""
        if self.mode == 'greater':
            result = tensor > self.threshold
        elif self.mode == 'smaller':
            result = tensor < self.threshold
        elif self.mode == 'greater_equal':
            result = tensor >= self.threshold
        elif self.mode == 'smaller_equal':
            result = tensor <= self.threshold
        else:
            raise NotImplementedError
        return result.float()


def pca(embedding, output_dimensions=3, reference=None, center_data=False):
    """
    Principal component analysis wrapping :class:`sklearn.decomposition.PCA`.
    Dimension 1 of the input embedding is reduced

    Parameters
    ----------
    embedding : torch.Tensor
        Embedding whose dimensions will be reduced.
    output_dimensions : int, optional
        Number of dimension to reduce to.
    reference : torch.Tensor, optional
        Optional tensor that will be used to train PCA on.
    center_data : bool, optional
        Whether to subtract the mean before PCA.

    Returns
    -------
        torch.Tensor
    """
    # embedding shape: first two dimensions correspond to batchsize and embedding(==channel) dim,
    # so shape should be (B, C, H, W) or (B, C, D, H, W).
    _pca = PCA(n_components=output_dimensions)
    # reshape embedding
    output_shape = list(embedding.shape)
    output_shape[1] = output_dimensions
    flat_embedding = embedding.cpu().numpy().reshape(embedding.shape[0], embedding.shape[1], -1)
    flat_embedding = flat_embedding.transpose((0, 2, 1))
    if reference is not None:
        # assert reference.shape[:2] == embedding.shape[:2]
        flat_reference = reference.cpu().numpy().reshape(reference.shape[0], reference.shape[1], -1)\
            .transpose((0, 2, 1))
    else:
        flat_reference = flat_embedding

    if center_data:
        means = np.mean(flat_reference, axis=0, keepdims=True)
        flat_reference -= means
        flat_embedding -= means

    pca_output = []
    for flat_reference, flat_image in zip(flat_reference, flat_embedding):
        # fit PCA to array of shape (n_samples, n_features)..
        _pca.fit(flat_reference)
        # ..and apply to input data
        pca_output.append(_pca.transform(flat_image))

    return torch.stack([torch.from_numpy(x.T) for x in pca_output]).reshape(output_shape)


# TODO: make PcaVisualizer take one embedding to fit and one to transform
class PcaVisualizer(BaseVisualizer):
    """
    PCA Visualization of high dimensional embedding tensor. An arbitrary number of channels is reduced
    to a multiple of 3 which are interpreted as sets RGB images.

    Parameters
    ----------
    n_components : int, optional
        Number of components to use. Must be divisible by 3.
    joint_specs: :obj:`tuple` of :obj:`str`, optional
        Entries only separated along these axis are treated jointly.

        Defaults to spatial dimensions.

        Use e.g. :code:`('B', 'H', 'W')` to run PCA jointly on all images of the batch.
        #TODO: make this example work. Right now, all dimensions except 'B' work.
    **super_kwargs

    """
    def __init__(self, n_components=3, joint_specs=('D', 'H', 'W'), **super_kwargs):
        super(PcaVisualizer, self).__init__(
            in_specs={'embedding': ['B', 'C'] + list(joint_specs)},
            out_spec=['B', 'C', 'Color'] + list(joint_specs),
            **super_kwargs)

        assert n_components % 3 == 0, f'{n_components} is not divisible by 3.'
        self.n_images = n_components // 3

    def visualize(self, embedding, **_):
        """"""
        # if there are not enough channels, add some zeros
        if embedding.shape[1] < 3 * self.n_images:
            expanded_embedding = torch.zeros(embedding.shape[0], 3 * self.n_images, *embedding.shape[2:])\
                .float().to(embedding.device)
            expanded_embedding[:, :embedding.shape[1]] = embedding
            embedding = expanded_embedding
        result = pca(embedding, output_dimensions=3 * self.n_images)
        result = result.contiguous().view((result.shape[0], self.n_images, 3) + result.shape[2:])
        return result


class MaskedPcaVisualizer(BaseVisualizer):
    """
    Version of PcaVisualizer that allows for an ignore mask. Data points which are labeled with :code:`ignore_label` in
    the segmentation are ignored in the PCA analysis.

    Parameters
    ----------
    ignore_label : int or float, optional
        Data points with this label in the segmentation are ignored.
    n_components : int, optional
        Number of components for PCA. Has to be divisible by 3, such that a whole number of RGB images can be
        returned.
    background_label : float, optional
        As in BaseVisualizer, here used by default to color the ignored region.
    **super_kwargs
    
    """
    def __init__(self, ignore_label=None, n_components=3, background_label=0, **super_kwargs):
        super(MaskedPcaVisualizer, self).__init__(
            in_specs={'embedding': 'BCDHW', 'segmentation': 'BCDHW'},
            out_spec=['B', 'C', 'Color', 'D', 'H', 'W'],
            background_label=background_label,
            **super_kwargs)
        self.ignore_label = ignore_label
        assert n_components % 3 == 0, f'{n_components} is not divisible by 3.'
        self.n_images = n_components // 3

    def visualize(self, embedding, segmentation, **_):
        """"""
        # if there are not enough channels, add some zeros
        if embedding.shape[1] < 3 * self.n_images:
            expanded_embedding = torch.zeros(embedding.shape[0], 3 * self.n_images, *embedding.shape[2:])\
                .float().to(embedding.device)
            expanded_embedding[:, :embedding.shape[1]] = embedding
            embedding = expanded_embedding

        if self.ignore_label is None:
            mask = torch.ones((embedding.shape[0],) + embedding.shape[2:])
        else:
            mask = segmentation != self.ignore_label
        if len(mask.shape) == len(embedding.shape):
            assert mask.shape[1] == 1, f'{mask.shape}'
            mask = mask[:, 0]
        mask = mask.bool()

        masked = [embedding[i, :, m] for i, m in enumerate(mask)]
        masked = [None if d.nelement() < self.n_images * 3 else pca(d[None], 3 * self.n_images, center_data=True)[0]
                  for d in masked]
        output_shape = list(embedding.shape)
        output_shape[1] = 3 * self.n_images
        result = torch.zeros(output_shape)
        for i, m in enumerate(mask):
            if masked[i] is not None:
                result[i, :, m] = masked[i]
        result = result.contiguous().view((result.shape[0], self.n_images, 3) + result.shape[2:])
        return result


class TsneVisualizer(BaseVisualizer):
    """
    tSNE Visualization of high dimensional embedding tensor. An arbitrary number of channels is reduced
    to a multiple of 3 which are interpreted as sets RGB images.

    Parameters
    ----------
    n_components : int, optional
        Number of components to use. Must be divisible by 3.
    joint_dims: :obj:`tuple` of :obj:`str`, optional
        Entries only separated along these axis are treated jointly.

        Defaults to spatial dimensions.
    **super_kwargs

    """
    def __init__(self, joint_dims=None, n_components=3, **super_kwargs):
        joint_dims = ['D', 'H', 'W'] if joint_dims is None else joint_dims
        assert 'C' not in joint_dims
        super(TsneVisualizer, self).__init__(
            in_specs={'embedding': joint_dims + ['C']},
            out_spec=joint_dims + ['C', 'Color'],
            **super_kwargs
        )
        assert n_components % 3 == 0, f'{n_components} is  not divisible by 3.'
        self.n_images = n_components // 3

    def visualize(self, embedding, **_):
        """"""
        shape = embedding.shape
        # bring embedding into shape (n_samples, n_features) as requested by TSNE
        embedding = embedding.contiguous().view(-1, shape[-1])

        result = TSNE(n_components=self.n_images * 3).fit_transform(embedding.cpu().numpy())
        result = torch.Tensor(result).float().to(embedding.device)
        # revert flattening, add color dimension
        result = result.contiguous().view(*shape[:-1], self.n_images, 3)
        return result


class UmapVisualizer(BaseVisualizer):
    """
    UMAP Visualization of high dimensional embedding tensor. An arbitrary number of channels is reduced
    to 3 which are interpreted as RGB.

    For a detailed discussion of parameters, see https://umap-learn.readthedocs.io/en/latest/parameters.html.

    Parameters
    ----------
    joint_dims: :obj:`tuple` of :obj:`str`, optional
        Entries only separated along these axis are treated jointly.

        Defaults to spatial dimensions.
    n_components : int, optional
        Number of components to use. Must be divisible by 3.
    n_neighbors: int, optional
        controls how many neighbors are considered for distance
        estimation on the manifold. Low number focuses on local
        distance, large numbers more on global structure, default 15.
    min_dist: float, optional
        minimum distance of points after dimension reduction, default 0.1.

    **super_kwargs

    """
    def __init__(self, joint_dims=None, n_components=3, n_neighbors=15, min_dist=0.1, **super_kwargs):
        assert umap_available, "You tried to use the UmapVisualizer without having UMAP installed."
        joint_dims = ['D', 'H', 'W'] if joint_dims is None else joint_dims
        assert 'C' not in joint_dims
        super(UmapVisualizer, self).__init__(
            in_specs={'embedding': joint_dims + ['C']},
            out_spec=joint_dims + ['C', 'Color'],
            **super_kwargs
        )

        self.min_dist = min_dist
        self.n_neighbors = n_neighbors

        assert n_components % 3 == 0, f'{n_components} is not divisible by 3.'
        self.n_images = n_components // 3

    def visualize(self, embedding, **_):
        """"""
        shape = embedding.shape
        # bring embedding into shape (n_samples, n_features) as requested by TSNE
        embedding = embedding.contiguous().view(-1, shape[-1])

        result = umap.UMAP(n_components=self.n_images * 3,
                           min_dist=self.min_dist,
                           n_neighbors=self.n_neighbors).fit_transform(embedding.cpu().numpy())
        result = torch.Tensor(result).float().to(embedding.device)
        # revert flattening, add color dimension
        result = result.contiguous().view(*shape[:-1], self.n_images, 3)
        return result


class NormVisualizer(BaseVisualizer):
    """
    Visualize the norm of a tensor, along a given direction (by default over the channels).

    Parameters
    ----------
    order : int, optional
        Order of the norm (Default is 2, euclidean norm).
    dim : str, optional
        Name of the dimension in which the norm is computed.
    **super_kwargs

    """
    def __init__(self, order=2, dim='C', **super_kwargs):
        super(NormVisualizer, self).__init__(
            in_specs={'tensor': ['B'] + [dim]},
            out_spec='B',
            **super_kwargs
        )
        self.order = order

    def visualize(self, tensor, **_):
        """"""
        return tensor.norm(p=self.order, dim=1)


class DiagonalSplitVisualizer(BaseVisualizer):
    """
    Combine two input images, displaying one above and one below the diagonal.

    Parameters
    ----------
    offset : int, optional
        The diagonal along which the image will be split is shifted by offset.
    **super_kwargs

    """
    def __init__(self, offset=0, **super_kwargs):
        super(DiagonalSplitVisualizer, self).__init__(
            in_specs={'upper_right_image': ['B', 'H', 'W'],
                      'lower_left_image': ['B', 'H', 'W']},
            out_spec=['B', 'H', 'W'],
            **super_kwargs
        )
        self.offset = offset

    def visualize(self, upper_right_image, lower_left_image, **_):
        """"""
        # upper_right and lower_left are tensors with shape (B, H, W)

        image_shape = upper_right_image.shape[1:]

        # construct upper triangular mask
        upper_right_mask = torch.ones(image_shape).triu(self.offset).float()

        upper_right_image = upper_right_image.float()
        lower_left_image = lower_left_image.float()
        return upper_right_image * upper_right_mask + lower_left_image * (1 - upper_right_mask)


class CrackedEdgeVisualizer(BaseVisualizer):
    """
    Visualize the boundaries of a segmentation.

    Parameters
    ----------
    width : int, optional
        width of the boundary in every direction
    connective_dims : tuple, optional
        Tuple of axis names. Edges in those axes will be shown.

        E.g. use :code:`('D', 'H', 'W')` to visualize edges in 3D.

    **super_kwargs

    """
    def __init__(self, width=1, connective_dims=('H', 'W'), **super_kwargs):
        self.connective_dims = list(connective_dims)
        super(CrackedEdgeVisualizer, self).__init__(
            in_specs={'segmentation': ['B'] + self.connective_dims},
            out_spec=['B'] + self.connective_dims,
            **super_kwargs
        )
        self.width = width
        self.pad_slice_tuples = self.make_pad_slice_tuples()

    def make_pad_slice_tuples(self):
        def make_tuple(offset):
            padding0 = [int(offset[i//2] if i % 2 == 0 else 0)
                        for i in reversed(range(2 * len(offset)))]
            padding1 = [int(offset[(i-1)//2] if i % 2 == 1 else 0)
                        for i in reversed(range(2 * len(offset)))]
            slicing = [slice(None), ] + [(slice(None) if off == 0 else slice((off)//2, -off//2))
                                         for off in offset]
            return tuple(padding0), tuple(padding1), tuple(slicing)

        offsets = np.eye(len(self.connective_dims)).astype(np.int32) * self.width
        return [make_tuple(list(offset)) for offset in offsets]

    def visualize(self, segmentation, **_):
        """"""
        directional_boundaries = []

        for padding0, padding1, slicing in self.pad_slice_tuples:
            # e.g. pad0 = (0, 0, 3, 0), pad1=(0, 0, 0, 3), slice = [..., 2:-1, :]
            padded0 = pad(segmentation, padding0)
            padded1 = pad(segmentation, padding1)
            directional_boundaries.append((padded0 != padded1)[slicing])
        return torch.stack(directional_boundaries, dim=0).max(dim=0)[0].float()


def _upsample_axis(tensor, axis, factor):
    shape = tensor.shape
    tensor = tensor.unsqueeze(axis + 1)
    tensor = tensor.expand(shape[:axis+1] + (factor,) + shape[axis+1:])
    tensor = tensor.reshape(shape[:axis] + (shape[axis] * factor,) + shape[axis+1:])
    return tensor


class UpsamplingVisualizer(BaseVisualizer):
    """
    Upsample a tensor along a list of axis (specified via specs) to a specified shape, by a list of specified
    factors or the shape of a reference tensor (given as an optional argument to visualize).

    Parameters
    ----------
    specs : list of str
        Specs of the axes to upsample along.
    shape : None or int or list, optional
        Shape after upsampling.
    factors: None or int or list, optional
        Factors to upsample by.
    **super_kwargs

    """
    def __init__(self, specs, shape=None, factors=None, **super_kwargs):
        self.specs = list(specs)
        self.out_shape = [shape] * len(specs) if isinstance(shape, int) else shape
        self.factors = [factors] * len(specs) if isinstance(factors, int) else shape
        assert self.out_shape is None or self.factors is None, \
            f'Pleas specify at most one of shape and factors'
        self.from_reference = self.out_shape is None and self.factors is None
        super(UpsamplingVisualizer, self).__init__(
            in_specs={
                'tensor': ['B'] + self.specs,
                'reference': ['B'] + self.specs
            },
            out_spec=['B'] + self.specs,
            **super_kwargs
        )

    def visualize(self, tensor, reference=None, **_):
        """"""
        if self.from_reference:
            assert reference is not None, \
                f'Please supply a reference when neither upsampled shape nor upsampling factors are specified at init.'
            out_shape = reference.shape[1:]
        else:
            if self.out_shape is not None:
                out_shape = self.out_shape
            else:
                out_shape = [s * f for s, f in zip(tensor.shape[1:], self.factors)]
        out_shape = np.array(out_shape)
        in_shape = np.array(tensor.shape[1:])
        assert all(out_shape % in_shape == 0), f'Cannot upsample from {in_shape} to {out_shape}.'
        factors = (out_shape / in_shape).astype(int)
        for i, factor in enumerate(factors):
            tensor = _upsample_axis(tensor, i+1, factor)
        return tensor
