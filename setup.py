"""
firelight - A visualization library for PyTorch tensors.
"""

import setuptools
import os

def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()

setuptools.setup(
    name="firelight",
    author="Roman Remme",
    author_email="roman.remme@iwr.uni-heidelberg.de",
    description="A visualization library for PyTorch tensors.",
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/inferno-pytorch/firelight',
    version="0.2.1",
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
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ]
)
