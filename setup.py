from setuptools import setup, find_namespace_packages

setup(
    name='lqrlearning',
    packages=find_namespace_packages(include=['lqrlearning.*']),
    version='0.1',
    install_requires=[
        'torch',
        'tensorboard',
        'cvxpylayers',
        'slycot',
        'control',
        'numpy',
        'scipy',
        'matplotlib',
        'click',
        'colorama',
    ])
