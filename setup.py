from distutils.core import setup
import setuptools

with open("requirements.txt") as f:
    requirements = list(map(lambda x: x.strip(), f.read().strip().splitlines()))

setup(
    name="bnn_zoo",
    version="0.1",
    url="https://github.com/nipunbatra/bnn-zoo/tree/pip-install-try1"
    python_requires=">=3.6",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
