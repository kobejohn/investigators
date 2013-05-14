import distribute_setup
distribute_setup.use_setuptools()

from setuptools import setup, find_packages

setup(
    name='imagefinder',
    version='0.1.1',
    py_modules=['imagefinder'],
    install_requires=['numpy'],
    tests_require=['mock'],
    package_data={'': ['cv2.pyd', '*.png']},
    include_package_data=True,
    url='http://github.com/kobejohn/ImageFinder',
    license='MIT',
    author='KobeJohn',
    author_email='niericentral@gmail.com',
    description='Find one image within another.'
)
