import io

from setuptools import find_packages, setup

setup(
    name='projekt_bazy',
    version='1.0.0',
    url='https://github.com/satproject/neuralheuristicsforsat/',
    license='MIT',
    maintainer='SAT TEAM',
    long_description=readme,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'jupyterlab',
        'tensorflow==1.12',
        'sympy',
        'sklearn',
        'scipy',
        'matplotlib'
        'pandas',
        'python-sat',
        'google-api-python-client',
        'google-cloud-storage',
        'oauth2client',
        'tqdm'
    ],
)
