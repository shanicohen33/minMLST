import re
import codecs
from setuptools import setup, find_packages
from os.path import join, abspath, dirname


def get_version(*file_paths):
    with codecs.open(join(abspath(dirname(__file__)), *file_paths), 'r') as fp:
        file = fp.read()
    version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", file, re.M)
    if version:
        return version.group(1)
    raise RuntimeError("Version string wasn't found.")


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="minmlst",
    version=get_version("minmlst", "__init__.py"),
    author="Shani Cohen",
    author_email="shani.cohen.33@gmail.com",
    description="Machine-learning based minimal MLST scheme for bacterial strain typing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shanicohen33/minMLST",
    packages=find_packages(),
    install_requires=[
        'shap>=0.28.5',
        'xgboost>=0.82'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
)

