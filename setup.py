#!/usr/bin/env python3
"""Setup configuration for MLM package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlm",
    version="0.1.0",
    author="MLNomad",
    description="Modular Language Model training and evaluation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax[tpu]>=0.4.0",
        "flax>=0.8.0",
        "optax>=0.1.0",
        "orbax-checkpoint>=0.4.0",
        "tiktoken>=0.5.0",
        "datasets>=2.0.0",
        "wandb>=0.15.0",
        "mteb>=1.0.0",
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "nmn>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlm-train=mlm.cli:main",
        ],
    },
)