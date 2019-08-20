import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="minmlst-v2",
    version="0.0.1",
    author="Shani Cohen",
    author_email="shani.cohen.33@gmail.com",
    description="Machine-learning based minimal MLST scheme for bacterial strain typing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shanicohen33/minMLST",
    packages=setuptools.find_packages(),
    # install_requires=[
    #     'markdown',
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
)


# setuptools.setup(
#     name="minmlst-v2",
#     version="0.0.1",
#     author="Shani Cohen",
#     author_email="shani.cohen.33@gmail.com",
#     description="Machine-learning based minimal MLST scheme for bacterial strain typing",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/pypa/sampleproject",
#     packages=setuptools.find_packages(),
#     # install_requires=[
#     #     'markdown',
#     # ],
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: Microsoft :: Windows",
#     ],
# )
