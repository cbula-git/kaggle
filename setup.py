"""
Setup script for nfl_analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file) as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith('#') and not line.startswith('pytest')
    ]

setup(
    name="nfl-analysis",
    version="0.1.0",
    author="NFL Analysis Team",
    description="A Python package for analyzing NFL tracking data from Kaggle competitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kaggle",  # Update with actual URL
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "nfl-consolidate=scripts.consolidate:main",
            "nfl-explore=scripts.explore:main",
        ],
    },
)
