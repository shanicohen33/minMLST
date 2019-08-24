from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="minmlst",
    version="0.0.1",  # get from tools
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