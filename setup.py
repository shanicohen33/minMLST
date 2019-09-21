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
    long_description="minMLST is a machine-learning based methodology for identifying a minimal subset of genes " +
                     "that preserves high discrimination among bacterial strains. It combines well known " +
                     "machine-learning algorithms and approaches such as XGBoost, distance-based hierarchical " +
                     "clustering, and SHAP. \nminMLST quantifies the importance level of each gene in an MLST " +
                     "scheme and allows the user to investigate the trade-off between minimizing the number " +
                     "of genes in the scheme vs preserving a high resolution among strain types.\n\n " +
                     "See more information in [GitHub](https://github.com/shanicohen33/minMLST).",
    long_description_content_type="text/markdown",
    url="https://github.com/shanicohen33/minMLST",
    packages=find_packages(),
    install_requires=[
        'shap>=0.28.5',
        'xgboost>=0.82',
        'dill>=0.3.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.6',
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux"
    ],
)

