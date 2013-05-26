try:
    from setuptools import setup
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup


setup(
    name='investigators',
    version='0.1.4',
    py_modules=['distribute_setup'],
    packages=['investigators'],
    install_requires=['numpy', 'PIL'],
    tests_require=['mock', 'PIL'],
    package_data={'': ['cv2.pyd', '*.png']},
    include_package_data=True,
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
