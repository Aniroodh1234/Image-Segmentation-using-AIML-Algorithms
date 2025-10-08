from setuptools import setup, find_packages

setup(
    name="image_segmentation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-algorithm image segmentation framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-segmentation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
    ],
)