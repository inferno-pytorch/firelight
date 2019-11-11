import numpy as np
import torch
from copy import copy
from collections import OrderedDict

# in this library, 'spec' always stands for a list of dimension names.
# eg. ['B', 'C', 'H', 'W'] standing for [Batch, Channel, Height, Width]


def join_specs(*specs):
    """
    Returns a list of dimension names which includes each dimension in any of the supplied specs exactly once, ordered
    by their occurrence in specs.

    Parameters
    ----------
    specs : list
        List of lists of dimension names to be joined

    Returns
    -------
    list

    Examples
    --------

    >>> join_specs(['B', 'C'], ['B', 'H', 'W'])
    ['B', 'C', 'H', 'W']
    >>> join_specs(['B', 'C'], ['H', 'B', 'W'])
    ['B', 'C', 'H', 'W']

    """
    if len(specs) != 2:
        return join_specs(specs[0], join_specs(*specs[1:]))
    spec1, spec2 = specs
    result = copy(spec1)
    for d in spec2:
        if d not in result:
            result.append(d)
    return result


def extend_dim(tensor, in_spec, out_spec, return_spec=False):
    """
    Adds extra (length 1) dimensions to the input tensor such that it has all the dimensions present in out_spec.

    Parameters
    ----------
    tensor : torch.Tensor
    in_spec : list
        spec of the input tensor
    out_spec : list
        spec of the output tensor
    return_spec : bool, optional
        Weather the output should consist of a tuple containing the output tensor and the resulting spec, or only the
        former.

    Returns
    -------
    torch.Tensor or tuple

    Examples
    --------

    >>> tensor, out_spec = extend_dim(
    ...     torch.empty(2, 3),
    ...     ['A', 'B'], ['A', 'B', 'C', 'D'],
    ...     return_spec=True
    ... )
    >>> print(tensor.shape)
    torch.Size([2, 3, 1, 1])
    >>> print(out_spec)
    ['A', 'B', 'C', 'D']

    """
    assert all(d in out_spec for d in in_spec)
    i = 0
    for d in out_spec:
        if d not in in_spec:
            tensor = tensor.unsqueeze(i)
        i += 1
    if return_spec:
        new_spec = out_spec + [d for d in in_spec if d not in out_spec]
        return tensor, new_spec
    else:
        return tensor


def moving_permutation(length, origin, goal):
    """
    Returns a permutation moving the element at position origin to the position goal (in the format requested by
    torch.Tensor.permute)

    Parameters
    ----------
    length : int
        length of the sequence to be permuted
    origin : int
        position of the element to be moved
    goal : int
        position the element should end up after the permutation

    Returns
    -------
    :obj:`list` of :obj:`int`

    Examples
    --------

    >>> moving_permutation(length=5, origin=1, goal=3)
    [0, 2, 3, 1, 4]
    >>> moving_permutation(length=5, origin=3, goal=1)
    [0, 3, 1, 2, 4]

    """
    result = []
    for i in range(length):
        if i == goal:
            result.append(origin)
        elif (i < goal and i < origin) or (i > goal and i > origin):
            result.append(i)
        elif goal < i <= origin:
            result.append(i-1)
        elif origin <= i < goal:
            result.append(i+1)
        else:
            assert False
    return result


def collapse_dim(tensor, to_collapse, collapse_into=None, spec=None, return_spec=False):
    """
    Reshapes the input tensor, collapsing one dimension into another. This is achieved by

    - first permuting the tensors dimensions such that the dimension to collapse is next to the one to collapse it into,
    - reshaping the tensor, making one dimension out of the to affected.

    Parameters
    ----------
    tensor : torch.Tensor
    to_collapse : int or str
        Dimension to be collapsed.
    collapse_into : int or str, optional
        Dimension into which the other will be collapsed.
    spec : list, optional
        Name of dimensions of input tensor. If not specified, will be taken to be range(len(tensor.shape())).
    return_spec : bool, optional
        Weather the output should consist of a tuple containing the output tensor and the resulting spec, or only the
        former.

    Returns
    -------
    torch.Tensor or tuple

    Examples
    --------

    >>> tensor = torch.Tensor([[1, 2, 3], [10, 20, 30]]).long()
    >>> collapse_dim(tensor, to_collapse=1, collapse_into=0)
    tensor([ 1,  2,  3, 10, 20, 30])
    >>> collapse_dim(tensor, to_collapse=0, collapse_into=1)
    tensor([ 1, 10,  2, 20,  3, 30])

    """
    spec = list(range(len(tensor.shape))) if spec is None else spec
    assert to_collapse in spec, f'{to_collapse}, {spec}'
    i_from = spec.index(to_collapse)
    if collapse_into is None:
        i_delete = i_from
        assert tensor.shape[i_delete] == 1, f'{to_collapse}, {tensor.shape[i_delete]}'
        tensor = tensor.squeeze(i_delete)
    else:
        assert collapse_into in spec, f'{collapse_into}, {spec}'
        i_to = spec.index(collapse_into)
        if i_to != i_from:
            i_to = i_to + 1 if i_from > i_to else i_to
            tensor = tensor.permute(moving_permutation(len(spec), i_from, i_to))
            new_shape = tensor.shape[:i_to-1] + (tensor.shape[i_to-1] * tensor.shape[i_to],) + tensor.shape[i_to+1:]
            tensor = tensor.contiguous().view(new_shape)
        else:
            i_from = -1  # suppress deletion of spec later
    if return_spec:
        new_spec = [spec[i] for i in range(len(spec)) if i is not i_from]
        return tensor, new_spec
    else:
        return tensor


def convert_dim(tensor, in_spec, out_spec=None, collapsing_rules=None, uncollapsing_rules=None,
                return_spec=False, return_inverse_kwargs=False):
    """
    Convert the dimensionality of tensor from in_spec to out_spec.

    Parameters
    ----------
    tensor : torch.Tensor
    in_spec : list
        Name of dimensions of the input tensor.
    out_spec : list, optional
        Name of dimensions that the output tensor will have.
    collapsing_rules : :obj:`list` of :obj:`tuple`, optional
        List of two element tuples. The first dimension in a tuple will be collapsed into the second (dimensions given
        by name).
    uncollapsing_rules : :obj:`list` of :obj:`tuple`, optional
        List of three element tuples. The first element of each specifies the dimension to 'uncollapse' (=split into
        two). The second element specifies the size of the added dimension, and the third its name.
    return_spec : bool, optional
        Weather the output should consist of a tuple containing the output tensor and the resulting spec, or only the
        former.
    return_inverse_kwargs : bool, optional
        If true, a dictionary containing arguments to reverse the conversion (with this function) are added to the
        output tuple.

    Returns
    -------
    torch.Tensor or tuple

    Examples
    --------

    >>> tensor = torch.Tensor([[1, 2, 3], [10, 20, 30]]).long()
    >>> convert_dim(tensor, ['A', 'B'], ['B', 'A'])  # doctest: +NORMALIZE_WHITESPACE
    tensor([[ 1, 10],
            [ 2, 20],
            [ 3, 30]])
    >>> convert_dim(tensor, ['A', 'B'], collapsing_rules=[('A', 'B')])  # doctest: +NORMALIZE_WHITESPACE
    tensor([ 1, 10,  2, 20,  3, 30])
    >>> convert_dim(tensor, ['A', 'B'], collapsing_rules=[('B', 'A')])  # doctest: +NORMALIZE_WHITESPACE
    tensor([ 1,  2,  3, 10, 20, 30])
    >>> convert_dim(tensor.flatten(), ['A'], ['A', 'B'], uncollapsing_rules=[('A', 3, 'B')])  # doctest: +NORMALIZE_WHITESPACE
    tensor([[ 1,  2,  3],
            [10, 20, 30]])

    """
    assert len(tensor.shape) == len(in_spec), f'{tensor.shape}, {in_spec}'

    to_collapse = [] if collapsing_rules is None else [rule[0] for rule in collapsing_rules]
    collapse_into = [] if collapsing_rules is None else [rule[1] for rule in collapsing_rules]
    uncollapsed_dims = []

    temp_spec = copy(in_spec)
    # uncollapse as specified
    if uncollapsing_rules is not None:
        for rule in uncollapsing_rules:
            if isinstance(rule, tuple):
                rule = {
                    'to_uncollapse': rule[0],
                    'uncollapsed_length': rule[1],
                    'uncollapse_into': rule[2]
                }
            uncollapsed_dims.append(rule['uncollapse_into'])
            tensor, temp_spec = uncollapse_dim(tensor, spec=temp_spec, **rule, return_spec=True)

    # construct out_spec if not given
    if out_spec is None:
        # print([d for d in in_spec if d not in to_collapse], collapse_into, uncollapsed_dims)
        out_spec = join_specs([d for d in in_spec if d not in to_collapse], collapse_into, uncollapsed_dims)

    # bring tensor's spec in same order as out_spec, with dims not present in out_spec at the end
    joined_spec = join_specs(out_spec, in_spec)
    order = list(np.argsort([joined_spec.index(d) for d in temp_spec]))
    tensor = tensor.permute(order)
    temp_spec = [temp_spec[i] for i in order]

    # unsqueeze to match out_spec
    tensor = extend_dim(tensor, temp_spec, joined_spec)
    temp_spec = joined_spec

    # apply dimension collapsing rules
    inverse_uncollapsing_rules = []  # needed if inverse is requested
    if collapsing_rules is not None:
        # if default to collapse into is specified, add appropriate rules at the end
        if 'rest' in to_collapse:
            ind = to_collapse.index('rest')
            collapse_rest_into = collapsing_rules.pop(ind)[1]
            for d in temp_spec:
                if d not in out_spec:
                    collapsing_rules.append((d, collapse_rest_into))
        # do collapsing
        for rule in collapsing_rules:
            if rule[0] in temp_spec:
                inverse_uncollapsing_rules.append({
                    'to_uncollapse': rule[1],
                    'uncollapsed_length': tensor.shape[temp_spec.index(rule[0])],
                    'uncollapse_into': rule[0]
                })
                # print(f'{tensor.shape}, {temp_spec}, {out_spec}')
                tensor, temp_spec = collapse_dim(tensor, spec=temp_spec, to_collapse=rule[0], collapse_into=rule[1],
                                                 return_spec=True)

    # drop trivial dims not in out_spec
    for d in reversed(temp_spec):
        if d not in out_spec:
            tensor, temp_spec = collapse_dim(tensor, to_collapse=d, spec=temp_spec, return_spec=True)

    assert all(d in out_spec for d in temp_spec), \
        f'{temp_spec}, {out_spec}: please provide appropriate collapsing rules'
    tensor = extend_dim(tensor, temp_spec, out_spec)

    result = [tensor]
    if return_spec:
        result.append(temp_spec)
    if return_inverse_kwargs:
        inverse_kwargs = {
            'in_spec': out_spec,
            'out_spec': in_spec,
            'uncollapsing_rules': inverse_uncollapsing_rules[::-1]
        }
        result.append(inverse_kwargs)
    if len(result) == 1:
        return result[0]
    else:
        return result


def uncollapse_dim(tensor, to_uncollapse, uncollapsed_length, uncollapse_into=None, spec=None, return_spec=False):
    """
    Splits a dimension in the input tensor into two, adding a dimension of specified length.

    Parameters
    ----------
    tensor : torch.Tensor
    to_uncollapse : str or int
        Dimension to be split.
    uncollapsed_length : int
        Length of the new dimension.
    uncollapse_into : str or int, optional
        Name of the new dimension.
    spec : list, optional
        Names or the dimensions of the input tensor
    return_spec : bool, optional
        Weather the output should consist of a tuple containing the output tensor and the resulting spec, or only the
        former.

    Returns
    -------
    torch.Tensor or tuple

    Examples
    --------

    >>> tensor = torch.Tensor([1, 2, 3, 10, 20, 30]).long()
    >>> uncollapse_dim(tensor, 0, 3, 1)  # doctest: +NORMALIZE_WHITESPACE
    tensor([[ 1,  2,  3],
            [10, 20, 30]])
    """
    # puts the new dimension directly behind the old one
    spec = list(range(len(tensor.shape))) if spec is None else spec
    assert to_uncollapse in spec, f'{to_uncollapse}, {spec}'
    assert uncollapse_into not in spec, f'{uncollapse_into}, {spec}'
    assert isinstance(tensor, torch.Tensor), f'unexpected type: {type(tensor)}'
    i_from = spec.index(to_uncollapse)
    assert tensor.shape[i_from] % uncollapsed_length == 0, f'{tensor.shape[i_from]}, {uncollapsed_length}'
    new_shape = tensor.shape[:i_from] + \
                (tensor.shape[i_from]//uncollapsed_length, uncollapsed_length) + \
                tensor.shape[i_from + 1:]
    tensor = tensor.contiguous().view(new_shape)
    if return_spec:
        assert uncollapse_into is not None
        new_spec = copy(spec)
        new_spec.insert(i_from + 1, uncollapse_into)
        return tensor, new_spec
    else:
        return tensor


def add_dim(tensor, length=1, new_dim=None, spec=None, return_spec=False):
    """
    Adds a single dimension of specified length (achieved by repeating the tensor) to the input tensor.

    Parameters
    ----------
    tensor : torch.Tensor
    length : int
        Length of the new dimension.
    new_dim : str, optional
        Name of the new dimension
    spec : list, optional
        Names of dimensions of the input tensor
    return_spec : bool, optional
        If true, a dictionary containing arguments to reverse the conversion (with this function) are added to the
        output tuple.

    Returns
    -------
    torch.Tensor or tuple

    """
    tensor = tensor[None].repeat([length] + [1] * len(tensor.shape))
    if return_spec:
        return tensor, [new_dim] + spec
    else:
        return tensor


def equalize_specs(tensor_spec_pairs):
    """
    Manipulates a list of tensors such that their dimension names (including order of dimensions) match up.

    Parameters
    ----------
    tensor_spec_pairs : :obj:`list` of :obj:`tuple`
        List of two element tuples, each consisting of a tensor and a spec (=list of names of dimensions).

    Returns
    -------
    torch.Tensor

    """
    specs = [p[1] for p in tensor_spec_pairs]
    unified_spec = list(np.unique(np.concatenate(specs)))
    result = []
    for i, (tensor, spec) in enumerate(tensor_spec_pairs):
        result.append(convert_dim(tensor, spec, unified_spec, return_spec=True))
    return result


def equalize_shapes(tensor_spec_pairs):
    """
    Manipulates a list of tensors such that their shapes end up equal.

    Axes that are not present in all tensors will be added as a trivial dimension to all tensors that do not have them.

    If shapes do not match along a certain axis, the tensors with the smaller shape will be repeated along that axis.
    Hence, the maximum length along each axis present in the list of tensors must be divisible by the lengths of all
    other input tensors along that axis.

    Parameters
    ----------
    tensor_spec_pairs : :obj:`list` of :obj:`tuple`
        List of two element tuples, each consisting of a tensor and a spec (=list of names of dimensions).

    Returns
    -------
    torch.Tensor

    """
    tensor_spec_pairs = equalize_specs(tensor_spec_pairs)
    unified_shape = np.max(np.array([list(p[0].shape) for p in tensor_spec_pairs]), axis=0)
    result = []
    for i, (tensor, spec) in enumerate(tensor_spec_pairs):
        old_shape = tensor.shape
        assert all(new_length % old_length == 0 for new_length, old_length in zip(unified_shape, old_shape)), \
            f'Shapes not compatible: {unified_shape}, {old_shape} (spec: {spec})'
        repeats = [new_length // old_length for new_length, old_length in zip(unified_shape, old_shape)]
        result.append((tensor.repeat(repeats), spec))
    return result


class SpecFunction:
    """
    Class that wraps a function, specified in the method :meth:`internal`, to be applicable to tensors with of almost
    arbitrary dimensionality. This is achieved by applying the following steps when the function is called:

    - The inputs are reshaped and their dimensions are permuted to match their respective order of dimensions
      specified in in_specs. Dimensions present in inputs but not requested by in_specs are collapsed in the
      batch dimension, labeled 'B' (per default, see collapse_into). Dimensions not present in the inputs but
      requested by in_specs are added (with length 1).

    - If the batch dimension 'B' is present in the in_specs, 'internal' is applied on the inputs, returning
      a tensor with dimensions as specified in out_spec.
      If 'B' is not present in the in_specs, this dimension is iterated over and each slice is individually
      passed through 'internal'. The individual outputs are then stacked, recovering the 'B' dimension.

    - Finally, the output is reshaped. The dimensions previously collapsed into 'B' are uncollapsed, and
      dimensions added in the first step are removed.

    Parameters
    ----------
    in_specs : dict, optional
        Dictionary specifying how the dimensionality and order of dimensions of input arguments of :meth:`internal`
        should be adjusted.

        - Keys: Names of input arguments (as in signature of :meth:`internal`)

        - Values: List of dimension names. The tensor supplied to internal under the name of the corresponding key
          will have this order of dimensions.

    out_spec : list, optional
        List of dimension names of the output of :meth:`internal`
    collapse_into : list, optional
        If given, the default behaviour of collapsing any extra given dimensions of states into the batch dimension
        'B' is overridden. Each entry of collapse_into must be a two element tuple, with the first element being the
        dimension to collapse, the second one being the dimension to collapse it into (prior to passing the tensor
        to :meth:`internal` ).
    suppress_spec_adjustment : bool, optional
        Argument to completely suppress the adjustment of dimensionalities in call(), for example if it is taken
        care of in call() of derived class (see firelight.visualizers.base.ConatainerVisualizer)

    """
    def __init__(self, in_specs=None, out_spec=None, collapse_into=None, suppress_spec_adjustment=True):
        if in_specs is None or out_spec is None:
            assert in_specs is None and out_spec is None, 'You probably want to supply both in_specs and an out_spec'
            assert suppress_spec_adjustment is True, 'You probably want to supply both in_specs and an out_spec'
            self.suppress_spec_adjustment = True
        else:
            self.suppress_spec_adjustment = False
            self.internal_in_specs = {key: list(value) for key, value in in_specs.items()}
            self.internal_out_spec = list(out_spec)
            assert (all('B' in spec for spec in self.internal_in_specs.values())) or \
                (all('B' not in spec for spec in self.internal_in_specs.values())), \
                f'"B" has to be in all or none of the internal specs: {self.internal_in_specs}'
            if all('B' not in spec for spec in self.internal_in_specs.values()):
                self.parallel = False
                self.internal_in_specs_with_B = {key: ['B'] + self.internal_in_specs[key] for key in in_specs}
            else:
                self.parallel = True
                self.internal_in_specs_with_B = self.internal_in_specs

        self.collapse_into = {'rest': 'B'} if collapse_into is None else collapse_into

    def __call__(self, *args, out_spec=None, return_spec=False, **kwargs):
        """
        Apply the wrapped function to a set of input arguments. Tensors will be reshaped as specified at initialization.

        Parameters
        ----------
        args : list
            List of positional input arguments to the wrapped function. They will be passed to :meth:`internal` without
            any processing.
        out_spec : list, optional
            List of dimension names of the output.
        return_spec : bool, optional
            Weather the output should consist of a tuple containing the output tensor and the resulting spec, or only the
            former.
        **kwargs
            Keyword arguments that will be passed to :meth:`internal`.
            The ones with names present in :paramref:`SpecFunction.in_specs` will be reshaped as required.

        Returns
        -------
        torch.Tensor or tuple

        """
        if self.suppress_spec_adjustment:  # just do internal if requested
            return self.internal(*args, out_spec=out_spec, return_spec=return_spec, **kwargs)

        given_spec_kwargs = [kw for kw in self.internal_in_specs if kw in kwargs]

        # determine the extra specs in the input. they will be put in the 'B' spec.
        extra_given_in_specs = OrderedDict()
        for kw in given_spec_kwargs:  # loop over given argument names that support dynamic specs
            assert len(kwargs[kw]) == 2, f'{kwargs[kw]}'  # has to be a pair of (arg, spec)
            arg, spec = kwargs[kw]
            kwargs[kw] = (arg, list(spec))  # make spec list, in case it is given as string
            # assert all(d in spec for d in extra_given_in_specs), \
            #     f'if extra specs are given, all input args need to have them: {kw}, {extra_given_in_specs}, {spec}'
            extra_given_in_specs.update({d: arg.shape[spec.index(d)] for d in spec
                                         if (d not in self.internal_in_specs[kw] and d not in extra_given_in_specs)})

        # print('extra specs', extra_given_in_specs)

        # add and repeat extra dimensions not present in some of the inputs
        for kw in given_spec_kwargs:
            arg, spec = kwargs[kw]
            for d in extra_given_in_specs:
                if d not in spec:
                    length = extra_given_in_specs[d]
                    arg, spec = add_dim(arg, length=length, new_dim=d, spec=spec, return_spec=True)
            kwargs[kw] = arg, spec

        # remove specs from extra given specs that are present in internal_in_specs
        # TODO: right now, this is unnecessary. allow for partially missing dims in the input_specs!
        for d in extra_given_in_specs:
            if not all(d not in spec for spec in self.internal_in_specs.values()):
                extra_given_in_specs.pop(d)
                assert d not in self.internal_out_spec, \
                    f'spec {d} is an internal_out_spec, cannot be an extra given spec'

        #if 'B' in extra_given_in_specs:
        #    del extra_given_in_specs['B']

        collapsing_rules = [(d, self.collapse_into.get(d, self.collapse_into.get('rest')))
                            for d in extra_given_in_specs]
        for kw in self.internal_in_specs:
            assert kw in kwargs, \
                f"Missing key '{kw}'. Provided keys were {kwargs.keys()} in SpecFunction of class {type(self)}"
            arg, spec = kwargs[kw]
            # make it so 'B' is present
            if 'B' not in spec:
                arg, spec = extend_dim(arg, spec, ['B'] + spec, return_spec=True)
            # collapse the extra dimensions of the input
            arg = convert_dim(arg, spec, self.internal_in_specs_with_B[kw], collapsing_rules)
            kwargs[kw] = arg  # finally update kwargs dictionary

        if self.parallel:
            result = self.internal(*args, **kwargs)
            spec = self.internal_out_spec
        else:
            n_batch = kwargs[list(self.internal_in_specs.keys())[0]].shape[0] if len(self.internal_in_specs) > 0 else 1
            result = torch.stack(
                [self.internal(*args, **{kw: kwargs[kw] if kw not in self.internal_in_specs else kwargs[kw][i]
                                         for kw in kwargs})
                 for i in range(n_batch)], dim=0)
            spec = ['B'] + self.internal_out_spec

        assert isinstance(result, torch.Tensor), f'unexpected type: {type(result)}'

        # uncollapse the previously collapsed dims
        dims_to_uncollapse = list(extra_given_in_specs.keys())
        for i in reversed(range(len(extra_given_in_specs))):
            d = dims_to_uncollapse[i]
            if d == 'B' and (d in self.internal_out_spec or not self.parallel):  # skip if function 'consumes' parallel dimension
                continue

            length = extra_given_in_specs[d]
            result, spec = uncollapse_dim(
                result,
                to_uncollapse=self.collapse_into.get(d, self.collapse_into.get('rest')),
                uncollapsed_length=length,
                uncollapse_into=d,
                spec=spec,
                return_spec=True
            )

        # finally, convert to out_spec, if specified
        if out_spec is not None:
            out_spec = list(out_spec)
            result, spec = convert_dim(result, in_spec=spec, out_spec=out_spec, return_spec=True)
        if return_spec:
            return result, spec
        else:
            return result

    def internal(self, *args, **kwargs):
        """
        Function that is being wrapped.
        """
        pass
