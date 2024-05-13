from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dyspyosis",
    version="0.0.2",
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
        "numpy>=1.26.3",
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "tensorflow==2.10.1",
        "keras==2.10.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0. https://creativecommons.org/licenses/by-nc-sa/4.0/",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
