try:
    from setuptools import setup
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup

setup(
    name='imagefinder',
    version='0.1.4.0',
    py_modules=['imagefinder', 'distribute_setup'],
    install_requires=['numpy'],
    tests_require=['mock'],
    package_data={'': ['cv2.pyd', '*.png']},
    include_package_data=True,
    url='http://github.com/kobejohn/ImageFinder',
    license='MIT',
    author='KobeJohn',
    author_email='niericentral@gmail.com',
    description='Find one image within another.',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Programming Language :: Python :: 2.7']
)
