from setuptools import setup

setup(
    author='Judit Acs',
    author_email='judit@mokk.bme.hu',
    name='dsl',
    provides=['dsl'],
    url='https://github.com/juditacs/dsl',
    packages=['dsl', 'dsl.representation', 'dsl.features', 'dsl.utils'],
    package_dir={'': '.'},
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=['numpy', 'scipy', 'scikit-learn'],
)
