from setuptools import setup, find_packages

setup(

    name = "harmonization-env",
    version = "1.0",
    author = "Begotti Pietro, Bianchi Luigi Amedeo, Cordoni Francesco Giuseppe",
    author_email = "pietro.begotti@studenti.unitn.it",

    description = "Harmonization environment implementation via reinforcement learning",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/pietrobegotti/harmonization",
   
    packages = find_packages(),

    include_package_data = True,
    package_data = {
        "harmonization_env.resources": ["*.sf2", "*.pth"],  # Include soundfont and parameter files
    },

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.6",
    install_requires = [
        "torch>=2.5.1",
        "numpy<2.1,>=1.22",
        "midiutil>=1.2.1",
        "midi2audio>=0.1.1", 
        "synthviz>=0.0.2",
        "tqdm"
    ],
)
