from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str)->List[str]:
    """
       this function will return list of requirements 
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='MHT-CET',
    version='0.0.1',
    author='Harsh',
    author_email='harshpatil3.1428@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)