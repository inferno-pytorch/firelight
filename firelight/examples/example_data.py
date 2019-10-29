import numpy as np
import torch
from skimage.data import binary_blobs
from skimage.filters import gaussian


def get_example_states():
    # generate some toy foreground/background segmentation
    batchsize = 5  # we will only visualize 3 of the 5samples
    size = 128
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


if __name__ == '__main__':
    from firelight import get_visualizer
    import matplotlib.pyplot as plt

    # Load the visualizer, passing the path to the config file. This happens only once, at the start of training.
    visualizer = get_visualizer('./configs/example_config_0.yml')

    # Get the example state dictionary, containing the input, target, prediction.
    states = get_example_states()

    # Call the visualizer.
    image_grid = visualizer(**states)

    # Log your image however you want.
    plt.imsave('visualizations/example_visualization.png', image_grid.numpy())
