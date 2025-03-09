from setuptools import find_packages,setup

from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function reads a requirements.txt file and returns a list of dependencies.
    It also removes '-e .' if present, as it is used for editable installs and not an actual package dependency.
    '''
    requirements = []

    # Open and read the requirements file
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        
        # Remove newline characters from each requirement
        requirements = [req.replace("\n","") for req in requirements]
        
        # Remove '-e .' if it exists, since it's only needed for local development
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

# Setup configuration for the ML project package
setup(
    name = 'MLproject',
    version = '0.0.1',                          # Version of the package
    author = 'Shiva Prasad Naroju',
    author_email = 'shivanaroju26@gmail.com',   # Contact email of the author
    packages = find_packages(),                 # Automatically discovers all packages in the project
    install_requires = get_requirements('requirements.txt'), # Fetch dependencies from requirements.txt
    long_description = open("README.md", "r").read(),  # Reads description from README
    long_description_content_type = "text/markdown",
    url = "https://github.com/Shiva-Prasad-Naroju/Student-Performance-MLOps.git"
)
