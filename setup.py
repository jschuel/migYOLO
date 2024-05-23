from setuptools import setup, find_packages

setup(
    name='migYOLO',
    version='0.1.0',
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
    python_requires='>=3.7',
    install_requires=[
        # List your dependencies here
    ],
    include_package_data=True,
)
