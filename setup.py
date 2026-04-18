from setuptools import setup, find_packages

setup(
    name="ultralytics-multiagent",
    version="1.0.0",
    description="Multi-Agent Object Detection Training System",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.0.0",
        "openai>=1.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "rich>=13.0.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "multiagent=main:cli",
        ],
    },
)
