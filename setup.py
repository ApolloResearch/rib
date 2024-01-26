from pathlib import Path

from setuptools import find_packages, setup

if Path("requirements.txt").exists():
    requirements = Path("requirements.txt").read_text("utf-8").splitlines()
else:
    requirements = []

setup(
    name="rib",
    version="1.0",
    description="Library for the Rotation into the Interaction Basis (RIB) method.",
    long_description=Path("README.md").read_text("utf-8"),
    author="Dan Braun",
    author_email="dan@apolloresearch.ai",
    url="https://github.com/ApolloResearch/rib",
    packages=find_packages(include=["rib", "rib.*", "rib_scripts", "rib_scripts.*"]),
    install_requires=requirements,
    extras_require={
        "dev": [
            "black==23.10.1",
            "isort~=5.13.2",
            "mypy~=1.8.0",
            "pylint~=3.0.3",
            "pytest~=7.4.4",
            "types-PyYAML~=6.0.12.12",
            "types-tqdm~=4.66.0.20240106",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
