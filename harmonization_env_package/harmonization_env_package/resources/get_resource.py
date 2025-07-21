import importlib.resources as pkg_resources

# Function to access resource files
def get_resource_file(filename):
    
    with pkg_resources.files("harmonization_env.resources").joinpath(filename) as resource_path:
        return str(resource_path)