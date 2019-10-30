"""
Realistic Example
=================

A close-to-real-world example of how to use firelight.
"""

##############################################################################
#  First of all, let us get some mock data to visualize.
#  We generate the following tensors:
#
# - :code:`input` of shape :math:`(B, D, H, W)`, some noisy raw data,
# - :code:`target` of shape :math:`(B, D, H, W)`, the ground truth foreground
#   background segmentation,
# - :code:`prediction` of shape :math:`(B, D, H, W)`, the predicted foreground
#   probability,
# - :code:`embedding` of shape :math:`(B, D, C, H, W)`, a tensor with an
#   additional channel dimension, as for example intermediate activations of a
#   neural network.
#

import numpy as np
import torch
from skimage.data import binary_blobs
from skimage.filters import gaussian


def get_example_states():
    # generate some toy foreground/background segmentation
    batchsize = 5  # we will only visualize 3 of the 5samples
    size = 64
    target = np.stack([binary_blobs(length=size, n_dim=3, blob_size_fraction=0.25, volume_fraction=0.5, seed=i)
                       for i in range(batchsize)], axis=0).astype(np.float32)

    # generate toy raw data as noisy target
    sigma = 0.5
    input = target + np.random.normal(loc=0, scale=sigma, size=target.shape)

    # compute mock prediction as gaussian smoothing of input data
    prediction = np.stack([gaussian(sample, sigma=3, truncate=2.0) for sample in input], axis=0)
    prediction = 10 * (prediction - 0.5)

    # compute mock embedding (if you need an image with channels for testing)
    embedding = np.random.randn(prediction.shape[0], 16, *(prediction.shape[1:]))

    # put input, target, prediction in dictionary, convert to torch.Tensor, add dimensionality labels ('specs')
    state_dict = {
        'input': (torch.Tensor(input).float(), 'BDHW'),  # Dimensions are B, D, H, W = Batch, Depth, Height, Width
        'target': (torch.Tensor(target).float(), 'BDHW'),
        'prediction': (torch.Tensor(prediction).float(), 'BDHW'),
        'embedding': (torch.Tensor(embedding).float(), 'BCDHW'),
    }
    return state_dict


# Get the example state dictionary, containing the input, target, prediction.
states = get_example_states()

for name, (tensor, spec) in states.items():
    print(f'{name}: shape {tensor.shape}, spec {spec}')

##############################################################################
# The best way to construct a complex visualizer to show all the tensors in a
# structured manner is to use a configuration file.
#
# We will use the following one:
#
# .. literalinclude:: ../../examples/example_config_0.yml
#    :language: yaml
#
# Lets load the file and construct the visualizer using :code:`get_visualizer`:

from firelight import get_visualizer
import matplotlib.pyplot as plt

# Load the visualizer, passing the path to the config file. This happens only once, at the start of training.
visualizer = get_visualizer('example_config_0.yml')

##############################################################################
# Now we can finally apply it on out mock tensors to get the visualization

# Call the visualizer.
image_grid = visualizer(**states)

# Log your image however you want.
plt.figure(figsize=(10, 6))
plt.imshow(image_grid.numpy())
plt.show()


