from setuptools import setup, find_packages

setup(
    name="inkstone-auto",
    version="0.1.0",
    description="Intern InkStone Automation Suite & SDK",
    author="Alaqmar",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.28.0",
    ],
    entry_points={
        "console_scripts": [
            "inkstone=inkstone.cli:main",
        ],
    },
    python_requires=">=3.8",
)
