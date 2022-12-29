from setuptools import setup, find_packages

setup(
    name='electropy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'astropy',
        'matplotlib',
        'scipy',
        'numpy',
        'uproot',
        'pandas',
        'iminuit',
        'gammapy',
        # 'pyslalib',
        'cmasher'
    ],
    include_package_data=True,
    )
