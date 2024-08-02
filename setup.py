#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import io
import os

# Readme
with open("README.md") as readme_file:
    readme = readme_file.read()


def read_requirements(file_path):
    with io.open(file_path, encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip()]
    return requirements


# Get base dir
base_dir = os.path.abspath(os.path.dirname(__file__))

# Get requirements
requirements_path = os.path.join(base_dir, "requirements.txt")
requirements = read_requirements(requirements_path)

test_requirements = [
    "pytest>=3",
]

setup(
    author="Herman Ye",
    author_email="hermanye233@icloud.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="VR puppeteer for Auromix",
    install_requires=requirements,
    include_package_data=True,
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords="auro_puppeteer",
    name="auro_puppeteer",
    packages=find_packages(include=["auro_puppeteer", "auro_puppeteer.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Auromix/auro_puppeteer",
    version="0.0.1",
    zip_safe=False,
)
