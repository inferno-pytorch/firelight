import yaml


def yaml2dict(path):
    """
    Read a yaml file.

    Parameters
    ----------
    path : str or dict
        Path to the file. If :class:`dict`, will be returned as is.

    Returns
    -------
    dict

    """
    if isinstance(path, dict):
        # Forgivable mistake that path is a dict already
        return path
    with open(path, 'r') as f:
        readict = yaml.load(f)
    return readict