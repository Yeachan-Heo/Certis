import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Certis", # Replace with your own username
    version="0.0.2",
    author="Yeachan-Heo",
    author_email="rlstart@kakao.com",
    description="Certis, Backtesting For y'all",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yeachan-Heo/Certis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
