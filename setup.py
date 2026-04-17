from setuptools import setup, find_packages

setup(
    name='uav-pe',
    version='0.1.0',
    description='UAV pursuit-evasion training and evaluation framework',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.10',
)
