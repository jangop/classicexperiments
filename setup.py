"""Persistent and reproducible experimental pipelines for Machine Learning.

See:
https://github.com/jangop/classicexperiments
"""

import pathlib

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="classicexperiments",
    version="0.1.0-alpha1",
    author="Jan Philip Göpfert",
    author_email="janphilip@gopfert.eu",
    description="Persistent and reproducible experimental pipelines for Machine Learning",
    keywords="machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jangop/classicexperiments",
    project_urls={
        "Bug Reports": "https://github.com/jangop/classicexperiments/issues",
        "Source": "https://github.com/jangop/classicexperiments",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
    ],
    python_requires=">=3.9, <4",
    install_requires=["loguru", "scikit-learn", "numpy", "appdirs"],
    extras_require={
        "dev": ["check-manifest", "black", "pylint"],
        "test": ["coverage", "pytest", "black", "pylint"],
    },
)
