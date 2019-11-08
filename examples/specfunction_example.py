"""
SpecFunction Example
====================

An example demonstrating the functionality of the :class:`SpecFunction` class.
"""

import torch
import matplotlib.pyplot as plt
from firelight.utils.dim_utils import SpecFunction

##############################################################################
#  Let us define a function that takes in two arrays and masks one with the
#  other:
#


class MaskArray(SpecFunction):
    def __init__(self, **super_kwargs):
        super(MaskArray, self).__init__(
            in_specs={'mask': 'B', 'array': 'BC'},
            out_spec='BC',
            **super_kwargs
        )

    def internal(self, mask, array, value=0.0):
        # The shapes are
        #   mask: (B)
        #   array: (B, C)
        # as specified in the init.

        result = array.clone()
        result[mask == 0] = value

        # the result has shape (B, C), as specified in the init.
        return result


##############################################################################
#  We can now apply the function on inputs of arbitrary shape, such as images.
#  The reshaping involved gets taken care of automatically:
#

W, H = 20, 10
inputs = {
    'array': (torch.rand(H, W, 3), 'HWC'),
    'mask': (torch.randn(H, W) > 0, 'HW'),
    'value': 0,
    'out_spec': 'HWC',
}

maskArrays = MaskArray()
result = maskArrays(**inputs)
print('output shape:', result.shape)

plt.imshow(result)
