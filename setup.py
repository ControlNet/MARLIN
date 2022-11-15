import init
import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

requirements = []
with open("requirements.txt", "r") as file:
    for line in file:
        requirements.append(line.strip())


version = init.read_version()
init.write_version(version)

setuptools.setup(
    name="marlin_pytorch",
    version=version,
    author="ControlNet",
    author_email="smczx@hotmail.com",
    description="Official pytorch implementation for MARLIN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ControlNet/MARLIN",
    project_urls={
        "Bug Tracker": "https://github.com/ControlNet/MARLIN/issues",
        "Source Code": "https://github.com/ControlNet/MARLIN",
    },
    keywords=["deep learning", "pytorch", "AI"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", include=["marlin_pytorch", "marlin_pytorch.*"]),
    package_data={
        "marlin_pytorch": [
            "version.txt"
        ]
    },
    python_requires=">=3.6",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
)
