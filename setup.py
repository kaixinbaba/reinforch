import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="reinforch",
    version="0.1.0",
    url="https://github.com/kaixinbaba/reinforch",
    license='MIT',

    author="JunjieXun",
    author_email="452914639@qq.com",

    description="RL common framework implements by pytorch",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'gym>=0.12.1',
        'torch>=1.0.0',
        'tqdm>=4.31.1',
        'pytest>=3.6.4',
        'logbook',
        'numpy>=1.16.2',
        'fire>=0.1.3',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
