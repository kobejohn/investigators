from distutils.core import setup

setup(
    name='ImageFinder',
    version='0.1.0',
    packages=['imagefinder'],
    requires=['cv2', 'numpy'],
    tests_require=['mock, numpy'],
    url='http://github.com/kobejohn/ImageFinder',
    license='MIT',
    author='KobeJohn',
    author_email='niericentral@gmail.com',
    description='Repeatedly find a template within scenes.'
)
