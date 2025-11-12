from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dyspyosis",
    version="1.0.0",
    author="Sebastian Proost",
    author_email="sebastian.proost@gmail.com",
    description="Calculate dysbiosis scores using Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raeslab/dyspyosis/",
    project_urls={
        "Bug Tracker": "https://github.com/raeslab/dyspyosis/issues",
    },
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.5.0",
        "tensorflow>=2.16.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0. https://creativecommons.org/licenses/by-nc-sa/4.0/",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
