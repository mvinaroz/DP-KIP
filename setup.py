from setuptools import setup, find_packages

setup(
    name='DPKIP',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy', 'absl-py', 'jax', 'jaxlib', 'autodp', 'tensorflow-datasets', 'kymatio', 'flax'],
)
