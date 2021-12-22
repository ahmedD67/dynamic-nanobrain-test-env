#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dynamic-nanobrain',
    version='0.1.0',
    description='Simulation package for nanowire neurons',
    long_description=readme,
    author='David Winge',
    author_email='winge.david@gmail.com',
    url='https://github.com/DavidWinge/dynamic-nanobrain',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)