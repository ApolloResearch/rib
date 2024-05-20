from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(file_path):
    if Path(file_path).exists():
        return Path(file_path).read_text("utf-8").splitlines()
    return []


setup(
    name="rib",
    version="2.0.0",
    description="Library for the Rotation into the Interaction Basis (RIB) method.",
    author="Dan Braun",
    author_email="dan@apolloresearch.ai",
    url="https://github.com/ApolloResearch/rib",
    packages=find_packages(include=["rib", "rib.*", "rib_scripts", "rib_scripts.*"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": read_requirements("requirements-dev.txt")},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
