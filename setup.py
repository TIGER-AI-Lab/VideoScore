from setuptools import setup, find_packages

setup(
    name='mantisscore',
    version='0.0.1',
    description='',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://github.com/jdf-prog/many-image-qa',
    install_requires=[
        "mantis-vl"
    ],
    extras_require={
        "train": [
            "mantis-vl[train]",
        ],
        "eval": [
            "mantis-vl[eval]"
        ]
    }
)



# change it to pyproject.toml
# [build-system]
