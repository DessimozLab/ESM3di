#!/usr/bin/env python
"""Setup script for esm3di package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="esm3di",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ESM + PEFT LoRA for 3Di per-residue prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/esm3di",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.5.0",
        "accelerate>=0.20.0",
    ],
    entry_points={
        "console_scripts": [
            "esm3di-train=esm3di.esmretrain:main",
            "esm3di-makefoldseekdb=esm3di.fastas2foldseekdb:main",
            "esm3di-buildtrainingset=esm3di.build_trainingset:main",
            "esm3di-download-alphafold=esm3di.testdataset:main",
        ],
    },
)
