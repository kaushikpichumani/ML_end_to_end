# why do we need setup.py

# this will be  help  me in building the entire machine learning application as package and also help me in deploying in Pypi

from setuptools import find_packages, setup
# meta data about project
from typing import List
HYPER_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    with open('requirements.txt') as fileobj:
        requirements = fileobj.readlines()
        [req.replace('\n','') for req in requirements]
        if HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)
            

setup(
    name = 'mlproject'
    ,version='0.0.1',
    author='kalyan',
    author_email='kaushikpichumani@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

)