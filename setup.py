from setuptools import find_packages, setup

setup(
    name="delivery_time_prediction",
    version="0.0.1",
    author="CerenKayaAkgün",
    author_email="cerenkaya07@gmail.com",
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'seaborn']
)