from setuptools import setup, find_packages

setup(
    name="sae_canonical_units",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sae-lens",
        "nnsight",
        "nnterp",
        "transformer-lens",
        "datasets",
        "pandas",
        "matplotlib",
        "scipy",
    ],
)