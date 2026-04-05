from setuptools import setup, find_packages

setup(
    name="pathracer",
    version="0.9.0",
    description="Physics-based path racing simulator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="David Baum",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scikit-image>=0.21",
        "scipy>=1.10",
        "scikit-fmm>=2024.1",
        "Pillow>=10.0",
    ],
    entry_points={
        "console_scripts": [
            "pathracer=cli:main",
        ],
    },
)
