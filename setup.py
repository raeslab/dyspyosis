import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dyspyosis",
    version="0.0.1",
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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="dyspyosis"),
    python_requires=">=3.10",
)
