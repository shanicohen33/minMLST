import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="minmlst-v5",
    version="0.0.1",
    author="Shani Cohen",
    author_email="shani.cohen.33@gmail.com",
    description="Machine-learning based minimal MLST scheme for bacterial strain typing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shanicohen33/minMLST",
    packages=setuptools.find_packages(),
    install_requires=[
        'shap',
        # 'xgboost>=0.82',
        'pandas',
        'numpy',
        'scikit-learn'
        # 'pickle'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
)

# 'shap>=0.28.5',
# # 'xgboost>=0.82',
# 'pandas>=0.19.2',
# 'numpy>=1.16.2',
# 'scikit-learn>=0.20.1'