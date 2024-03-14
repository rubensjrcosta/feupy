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
    print(f"Python cript name: {script_name}.py")
    output_dir = f'{output_dir}/'
    print(f"Output directory: {mod_file}")
    return script_name, output_dir

# In[4]:



