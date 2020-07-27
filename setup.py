"""The setup script."""

from setuptools import find_packages, setup

requirements = [
    "numpy>=1.18.3",
    "jax>=0.1.65",
    "jaxlib>=0.1.45",
    "tqdm>=4.45.0",
    "pydantic>=1.5.1",
    "wandb>=0.9.4",
]

setup_requirements = ["pytest-runner", "setuptools>=38.6.0", "wheel>=0.31.0"]

test_requirements = ["pytest", "pytest-sugar"]

with open("README.md") as infile:
    long_description = infile.read()

setup(
    name="colin_net",
    version="0.0.1",
    description="Deep Learning Framework using Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Colin Sullivan",
    author_email="csulliva@brandeis.edu",
    url="https://github.com/niloch/colin_net",
    packages=find_packages(where="src", include=["colin_net"]),
    package_dir={"": "src"},
    package_data={"colin_net": ["py.typed"]},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.7",
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
