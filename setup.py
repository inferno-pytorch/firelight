"""
firelight - A visualization library for PyTorch tensors.
"""

import setuptools


setuptools.setup(
    name="firelight",
    author="Roman Remme",
    author_email="roman.remme@iwr.uni-heidelberg.de",
    description="A visualization library for PyTorch tensors.",
    version="0.2.0",
    install_requires=[
        "pyyaml>=3.12",
        "matplotlib",
        "numpy",
        "scikit-learn",
        "scikit-image",
        "torch",
    ],
    extras_requires={
    'umap': ['umap-learn>=0.3.8'],
    },
    license="Apache Software License 2.0",
    packages=setuptools.find_packages(),
)
