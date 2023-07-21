from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

required_packages = ["tensorboard",
    "Dassl.pytorch @ git+ssh://git@github.com:ManuelRoeder/Dassl.pytorch.git"
]

setup(
    name="FedMixStyle",
    version="0.1.0",
    description="Efficient Cross-Domain Federated Learning by MixStyle Approximation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ManuelRoeder/FedMixStyle",
    author="Manuel Roeder",
    author_email="mroeder57@gmail.com",
    license="MIT",
    #packages=["lightningflower"],
    install_requires=required_packages,
    #python_requires='>=3.8.12',
    package_data={"": ["README.md", "LICENSE"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)