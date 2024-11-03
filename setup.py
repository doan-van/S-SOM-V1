
from setuptools import setup, find_packages

setup(
    name="ssom",
    version="0.1.0",
    description="A simple Self-Organizing Map (SOM) library.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
