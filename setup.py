"""
firelight - A visualization library for PyTorch tensors.
"""

import setuptools


setuptools.setup(
    name="firelight",
    author="Roman Remme",
    author_email="roman.remme@iwr.uni-heidelberg.de",
    description="A visualization library for PyTorch tensors.",
    version="0.1",
    install_requires=[
        "scikit-image",
        "pyyaml>=3.12",
        "numpy",
        "scikit-learn",
        "torch",
    ],  # TODO
    packages=setuptools.find_packages(),
)
