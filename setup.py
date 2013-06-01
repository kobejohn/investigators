try:
    from setuptools import setup, find_packages
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages


setup(
    name='investigators',
    version='0.1.4.1',
    py_modules=['distribute_setup'],
    packages=find_packages(),
    package_data={'': ['*.pyd', '*.png']},
    install_requires=['numpy', 'PIL'],
    tests_require=['mock', 'numpy', 'PIL'],
    url='http://github.com/kobejohn/investigators',
    license='MIT',
    author='KobeJohn',
    author_email='niericentral@gmail.com',
    description='Achieve various investigations of data by combining'
                ' easy-to-use investigator classes.',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Programming Language :: Python :: 2.7',
        'Operating System :: Microsoft :: Windows']
)
