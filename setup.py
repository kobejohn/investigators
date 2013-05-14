from distutils.core import setup

setup(
    name='ImageFinder',
    version='0.1.1',
    packages=['imagefinder'],
    requires=['cv2', 'numpy'],
    tests_require=['cv2', 'mock', 'numpy'],
    url='http://github.com/kobejohn/ImageFinder',
    license='MIT',
    author='KobeJohn',
    author_email='niericentral@gmail.com',
    description='Find one image within another.'
)
