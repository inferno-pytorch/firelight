# Firelight

Firelight is a visualization library for pytorch. 
Its core object is a **visualizer**, which can be called passing a dictionary of states (such as `inputs`, `target`, 
`prediction`) returning a visualization of the data. What exactly that visualization shows, is specified in a .yml
configuration file.

Why you will like firelight initially:
- Neat Image Grids, lining up inputs, targets and predictions
- Colorful images: Automatic scaling for RGB, matplotlib colormaps for grayscale data, randomly colored label images
- Many available visualizers

Why you will keep using firelight:
- Everything in one config file
- Easily write your own visualizers
- Generality in Dimensions: All visualizers usable with data of arbitrary dimension

## Installation
On python 3.6+:

```bash
# Clone the repository
git clone https://github.com/imagirom/firelight
cd firelight/
# Install
python setup.py install
```

## Example

- Run the example `firelight/examples/example_data.py`
- Config file:

```yaml
RowVisualizer: # stack the outputs of child visualizers as rows of an image grid
  input_mapping:
    global: ['B': ':3', 'D': ':6:2'] # Show only 3 samples in each batch ('B'), and some slices along depth ('D').
    prediction: ['C': '0']  # Show only the first channel of the prediction

  pad_value: [0.2, 0.6, 1.0] # RGB color of separating lines
  pad_width: {B: 4, H: 0, W: 0, rest: 2} # Padding for batch ('B'), height ('H'), width ('W') and other dimensions.

  visualizers:
    # First row: Ground truth
    - IdentityVisualizer:
        input_mapping:
          tensor: 'target' # show the target

    # Second row: Raw input
    - IdentityVisualizer:
        input_mapping:
          tensor: ['input', C: '0'] # Show the first channel ('C') of the input.
        cmap: viridis  # Name of a matplotlib colormap.

    # Third row: Prediction with segmentation boarders on top.
    - OverlayVisualizer:
        visualizers:
          - CrackedEdgeVisualizer: # Show borders of target segmentation
              input_mapping:
                segmentation: 'target'
              width: 2
              opacity: 0.5 # Make output only partially opaque.
          - IdentityVisualizer: # prediction
              input_mapping:
                tensor: 'prediction'
              cmap: Spectral

    # Fourth row: Foreground probability, calculated by sigmoid on prediction
    - IdentityVisualizer:
        input_mapping:
          tensor: ['prediction', pre: 'sigmoid'] # Apply sigmoid function from torch.nn.functional before visualize.
        value_range: [0, 1] # Scale such that 0 is white and 1 is black. If not specified, whole range is used.
```
- Python code:

```python
from firelight import get_visualizer
import matplotlib.pyplot as plt

# Load the visualizer, passing the path to the config file. This happens only once, at the start of training.
visualizer = get_visualizer('./configs/example_config_0.yml')

# Get the example state dictionary, containing the input, target, prediction
states = get_example_states()

# Call the visualizer
image_grid = visualizer(**states)

# Log your image however you want
plt.imsave('visualizations/example_visualization.jpg', image_grid.numpy())
```
- Resulting visualization: 

![Example Image Grid](/firelight/examples/visualizations/example_visualization.png)
