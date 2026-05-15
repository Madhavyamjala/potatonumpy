"""Setup script for PotatoNumPy."""

from setuptools import setup, find_packages

setup(
    name="potatonumpy",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    author="Madhav",
    author_email="yamjalamadhav@users.noreply.github.com",
    description="A pure Python linear algebra and tensor library for educational purposes",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Madhavyamjala/potatonumpy",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
