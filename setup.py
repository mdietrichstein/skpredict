from setuptools import setup

setup(
    name="skpredict",
    install_requires={
        "numpy>=1.20"
    },
    extras_require={
        "scikit-learn": ["scikit-learn>=0.22.0"]
    }
)
