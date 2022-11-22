# coding: utf-8
from setuptools import setup

setup(name='torch_train_test_loop',
    version='1.0.0',
    description='Composable training/testing of PyTorch deep learning models with minimal overhead.',
    url='https://github.com/glassroom/torch_train_test_loop',
    author='Franz A. Heinsen',
    author_email='franz@glassroom.com',
    license='MIT',
    packages=['torch_train_test_loop'],
    install_requires='torch',
    zip_safe=False)
