from setuptools import setup, find_packages

setup(
    name='migYOLO',
    version='1.0.0',
    description='Tools for evaluating MIGDAL image data with a pre-trained version of YOLOv8',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jschuel/migYOLO',
    author='Jeff Schueler',
    author_email='schuel93@gmail.com',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.17.5",
        "pandas>=1.0.0",
        "matplotlib>=3.2.0",
        "tqdm>=4.41.1",
        "scikit-image>=0.18.0",
        "scipy>=1.5.0",
        "pillow>=7.0.0",
        "pyyaml>=5.3",
        "pyarrow>=1.0.0",
    ],
    include_package_data=True,
)
