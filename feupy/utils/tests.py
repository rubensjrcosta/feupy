from pathlib import Path

__all__ = [
    "get_names_tests",
]


def get_names_tests(mod_file=None):
    file_path = (Path(mod_file))
    module_path = file_path.parent
    output_dir = str(module_path)
    script_name = str(file_path)
    script_name = script_name.replace(f"{output_dir}/", "").replace(".py", "")
    print(f"Python script name: {script_name}.py")
    output_dir = f'{output_dir}/'
    print(f"Output directory: {mod_file}")
    return script_name, output_dir

# In[4]:




# # !pip install ipynbname
# import ipynbname
# from feupy.utils.tests import get_names_tests
# from feupy.analysis import config

# script_name, output_dir = get_names_tests(mod_file=config.__file__)

# !jupyter nbconvert test_simulation.ipynb --to notebook --nbformat 3

# !jupyter nbconvert test_simulation.ipynb --to  script --output "{script_name}" --output-dir '{output_dir}'