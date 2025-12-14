#!/usr/bin/env python3
"""
TSQVT: Twistorial Spectral Quantum Vacuum Theory
=================================================

A Python package for computing Standard Model parameters from spectral geometry.

Installation:
    pip install -e .

Usage:
    import tsqvt
    from tsqvt.core import SpectralManifold
    from tsqvt.gauge import compute_gauge_couplings
"""

from setuptools import setup, find_packages
import os

# Read version from package
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'tsqvt', '__init__.py')
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            with open(init_path, 'r', encoding=enc) as f:
                for line in f:
                    if line.startswith('__version__'):
                        return line.split('=')[1].strip().strip('"\'')
            break
        except UnicodeDecodeError:
            continue
    return '0.1.0'


# Read README for long description
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def get_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ['numpy>=1.20.0', 'scipy>=1.7.0', 'sympy>=1.9']

setup(
    name='tsqvt',
    version=get_version(),
    author='Kerym Makraini',
    author_email='kerym.makraini@example.com',
    description='Twistorial Spectral Quantum Vacuum Theory: SM parameters from spectral geometry',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/kerym-makraini/sm-parameters-repo',
    project_urls={
        'Documentation': 'https://github.com/kerym-makraini/sm-parameters-repo#readme',
        'Bug Reports': 'https://github.com/kerym-makraini/sm-parameters-repo/issues',
        'Source': 'https://github.com/kerym-makraini/sm-parameters-repo',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'docs']),
    python_requires='>=3.8',
    install_requires=get_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'nbsphinx>=0.8.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='physics, quantum gravity, spectral geometry, noncommutative geometry, twistor theory, standard model',
    entry_points={
        'console_scripts': [
            'tsqvt-predict=scripts.run_predictions:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
