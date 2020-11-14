import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datasets",
    version="1.0",
    author="Vaux Gomes",
    author_email="vauxgomes@gmail.com",
    description="Datasets for machine learning testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vauxgomes/datasets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
