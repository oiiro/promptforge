from setuptools import setup

setup(
    name="promptforge-cli",
    version="1.0.0",
    py_modules=["promptforge_cli"],
    install_requires=[
        "click>=8.1.7",
        "rich>=13.7.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0"
    ],
    entry_points={
        "console_scripts": [
            "promptforge=promptforge_cli:cli",
        ],
    },
)