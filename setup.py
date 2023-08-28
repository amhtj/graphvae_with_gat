from setuptools import setup, find_packages

name = "graphrnn"
setup(
    name=name,
    version='0.0.1',
    url="https://github.com/standford/graphrnn",
    packages=find_packages(exclude='tests'),
    install_requires=[],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
)
