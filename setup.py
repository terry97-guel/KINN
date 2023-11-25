from setuptools import setup, find_packages

setup(
    name='kinn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add any dependencies your package needs
    ],
    # Metadata
    author='Taerim Yoon',
    author_email='taerimyoon@korea.ac.kr',
    description='Kinematics Informed Neural Networks',
    long_description='To identify and control soft robots in a data-efficient manner',
    url='https://github.com/terry97-guel/KINN.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        # Add more classifiers as needed
    ],
)