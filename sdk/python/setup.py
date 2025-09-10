from setuptools import setup, find_packages

setup(
    name="promptforge-sdk",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
)