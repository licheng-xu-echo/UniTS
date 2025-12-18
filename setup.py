import os
from setuptools import setup,find_packages

setup(
    name="units",
    version="0.15.0",
    description="A universal diffusion model for transition state geometry generation",
    keywords=[],
    url="https://github.com/licheng-xu-echo/UniTS",
    author="Li-Cheng Xu",
    author_email="licheng_xu@zju.edu.cn",
    license="MIT License",
    packages=find_packages(),
    install_package_data=True,
    zip_safe=False,
    install_requires=[],
    package_data={"":["*.csv","*.pt"]},
)