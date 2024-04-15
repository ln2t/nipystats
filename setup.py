from setuptools import setup, find_packages

# Function to read the contents of the requirements file
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='nipystats',
    version='2.1.1',
    url='https://github.com/ln2t/nipystats',
    author='Antonin Rovai',
    author_email='antonin.rovai@hubruxelles.be',
    description='nilearn-based BIDS app for task-based fMRI data analysis (first and second level) ',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'nipystats = nipystats.nipystats:main',
        ]},
    include_package_data=True
)
