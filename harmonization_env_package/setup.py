from setuptools import setup, find_packages

setup(

    name = "harmonization_environment",

    version = "1.0",

    packages = find_packages(),

    install_requires = [
        "torch",
        "numpy",
        "midiutil",
        "midi2audio", 
        "synthviz",
        "tqdm"
    ],

    author = 'Begotti Pietro, Bianchi Amedeo Luigi, Cordoni Francesco Giuseppe'
)
