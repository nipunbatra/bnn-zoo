from distutils.core import setup
import setuptools

with open("requirements.txt") as f:
    requirements = list(map(lambda x: x.strip(), f.read().strip().splitlines()))

setup(
    name="bnn_zoo",
    version="0.1",
    description = "MLP(Multi-layer Perceptron) from scratch using JAX",
    url="https://github.com/nipunbatra/bnn-zoo",
    install_requires=requirements,
    packages=setuptools.find_packages(),
)
