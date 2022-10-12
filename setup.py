from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="Intersectional-Fairness",
    version="0.1.0",
    license="Apache-2.0",
    description="Intersectional Fairness (ISF) is a bias detection and mitigation technology for intersectional bias, which combinations of multiple protected attributes causes. ",
    author="Fujitsu Limited",
    author_email="fj-isf-contact@dl.jp.fujitsu.com",
    url="https://github.com/intersectional-fairness/isf.git",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt')
)
