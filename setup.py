from setuptools import setup

setup(
    name='penelytics',
    version='0.0.2',
    author='Jules Penel',
    author_email='julespenel@gmail.com',

    packages=['penelytics'],# + [f'fancychart\\{x}' for x in ['data', 'excel_interaction', 'templates']],
    # scripts=['bin/script1','bin/script2'],
    # url='http://pypi.python.org/pypi/PackageName/',
    license='LICENSE.txt',
    description='Advanced charting templates and some data analytics tool',
    long_description=open('README.txt').read(),
    install_requires=[
        "fancychart>=0.1.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.1",
        "pandas>=1.2.2",
    ],
)
