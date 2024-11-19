import os
import subprocess

# Get the path of this file
current_work_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current working directory: {current_work_dir}")

# Get which python path
python_path = subprocess.run(['which', 'python'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
python_path = '/'.join(python_path.split('/')[:-2]) + '/lib/python3.10/site-packages'
print(f"Python site-packages path: {python_path}")

# Function to create a backup of the existing target file
def backup_target_file(target_file_path):
    if os.path.islink(target_file_path):
        if not os.path.exists(os.readlink(target_file_path)):
            print(f"Found a broken symlink at {target_file_path}, removing it.")
            os.unlink(target_file_path)  # Remove broken symlink
        else:
            # Valid symlink
            print(f"Found a valid symlink at {target_file_path}, skipping backup.")
            return
    elif os.path.exists(target_file_path):
        # It's a regular file
        backup_path = target_file_path + '.bak'
        if os.path.exists(backup_path):
            print(f"Backup file {backup_path} already exists, skipping backup.")
        else:
            print(f"Backing up {target_file_path} to {backup_path}")
            os.rename(target_file_path, backup_path)
    else:
        # Does not exist
        pass  # Do nothing

# Function to create a symbolic link using `ln -s`
def create_symlink(source_file, target_file):
    target_dir = os.path.dirname(target_file)
    if not os.path.exists(target_dir):
        print(f"Creating directory {target_dir}")
        os.makedirs(target_dir)
    print(f"Creating symlink from {source_file} to {target_file}")
    subprocess.run(['ln', '-s', source_file, target_file], check=True)

# For each sublist, the first one is the file name under the current working directory,
# the second one is the target file path
env_changed_files = [
    [f'{current_work_dir}/modified_files/my_elastic_agent.py', f'{python_path}/deepspeed/elasticity/elastic_agent.py'],
    [f'{current_work_dir}/modified_files/my_modeling_flash_attention_utils.py', f'{python_path}/transformers/modeling_flash_attention_utils.py'],
    [f'{current_work_dir}/modified_files/my_ulysses_attn_layer.py', f'{python_path}/yunchang/ulysses/attn_layer.py'],
    [f'{current_work_dir}/modified_files/my_ulysses_init.py', f'{python_path}/yunchang/ulysses/__init__.py'],
    [f'{current_work_dir}/modified_files/my_import_utils.py', f'{python_path}/transformers/utils/import_utils.py'],
    [f'{current_work_dir}/modified_files/my_modeling_llama.py', f'{python_path}/transformers/models/llama/modeling_llama.py'],
    [f'{current_work_dir}/modified_files/my_utils_init.py', f'{python_path}/transformers/utils/__init__.py'],
    [f'{current_work_dir}/modified_files/my_transformers_init.py', f'{python_path}/transformers/__init__.py'],
    [f'{current_work_dir}/modified_files/my_modeling_rope_utils.py', f'{python_path}/transformers/modeling_rope_utils.py'],
    [f'{current_work_dir}/modified_files/my_modeling_qwen2.py', f'{python_path}/transformers/models/qwen2/modeling_qwen2.py'],
    [f'{current_work_dir}/modified_files/my_modeling_mistral.py', f'{python_path}/transformers/models/mistral/modeling_mistral.py'],
    [f'{current_work_dir}/modified_files/my_cache_utils.py', f'{python_path}/transformers/cache_utils.py'],
    [f'{current_work_dir}/modified_files/my_modeling_utils.py', f'{python_path}/transformers/modeling_utils.py'],
]

# Iterate over the list and perform the necessary operations
for file_pair in env_changed_files:
    source_file = file_pair[0]
    target_file = file_pair[1]

    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Source file {source_file} does not exist!")
        exit(1)

    # Check if target file exists or is a symlink
    if os.path.lexists(target_file):
        if os.path.islink(target_file):
            # Check if symlink is valid or broken
            if os.path.exists(target_file):
                print(f"Target file {target_file} is a valid symlink.")
                print(f"Unlinking the symlink {target_file}")
                os.unlink(target_file)
            else:
                print(f"Target file {target_file} is a broken symlink.")
                print(f"Removing the broken symlink {target_file}")
                os.unlink(target_file)
        else:
            # Regular file
            print(f"Target file {target_file} exists.")
    else:
        print(f"Target file {target_file} does not exist.")

    # Prompt the user
    response = input(f"Do you want to backup the target file and create a symlink for {target_file}? (y/n): ").strip().lower()
    if response == 'y':
        # Backup the existing target file if it exists
        backup_target_file(target_file)

        # Create a symbolic link
        create_symlink(source_file, target_file)
    else:
        print("Skipping this file.")
        continue