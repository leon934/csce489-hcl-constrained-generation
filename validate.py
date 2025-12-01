import subprocess
import os
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

def validate_terraform_files(path_to_terraform_files: Path) -> Path:
  '''
  validates whether the generated terraform files are syntactically valid

  note that the way this function works is a temporary directory is created for each .tf file. meaning that a similar process should be done for further validation down the line, such as setting up the temporary directory w/ "terraform init" and "terraform validate".
  
  args:
  - just requires a path to the generated terraform files within a specific directory
      - note that this is the default behavior based on my changes in benchmark.py

  returns:
  - path to new directory of the valid terraform files for further processing
  '''

  # gets list of all terraform files
  terraform_files = list(path_to_terraform_files.glob("*.tf"))

  if len(terraform_files) == 0:
    return

  # first run terraform init
  # "terraform init" isn't required to be ran if there are no new files TO configure, but it's wtv if we run it anyways
  valid_tf_file_paths = []

  # creates a temporary directory to test each file individually
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir_path = Path(temp_dir)

    for tf_file_path in tqdm(terraform_files, desc="Iterating through .tf files..."):
      # remove the file from temp directory to test next one
      for temp_file in temp_dir_path.iterdir():
        if temp_file.is_file() or temp_file.is_symlink():
            temp_file.unlink()
        elif temp_file.is_dir():
            shutil.rmtree(temp_file)

      # copy the file into temp directory
      shutil.copy(tf_file_path, temp_dir_path)

      # runs "terraform init" and checks error code to see if it ran properly
      # -backend=false prevents command from trying to connect to backend, which might cause the model to fail
      init_proc = subprocess.run(
        ["terraform", "init", "-backend=false"], 
        cwd=str(temp_dir_path), 
        capture_output=True, 
        text=True
      )
      
      # retcode = 0 means success
      if init_proc.returncode != 0:
        continue

      # we then move onto "terraform validate" if no errors occurred
      validate_proc = subprocess.run(
        ["terraform", "validate"],
        cwd=str(temp_dir_path),
        capture_output=True,
        text=True
      )

      if validate_proc.returncode == 0:
        valid_tf_file_paths.append(tf_file_path)

  valid_directory_path = Path(f"{path_to_terraform_files.parent}/valid_outputs")

  # deletes all existing files in valid_outputs directory from potentially previous iterations
  if os.path.exists(valid_directory_path):
    shutil.rmtree(valid_directory_path)
  
  os.makedirs(valid_directory_path, exist_ok=True)

  for terraform_file_path in valid_tf_file_paths:
    shutil.copy(terraform_file_path, valid_directory_path)

  # once we have the files that pass "terraform init", we then run the command again so it properly is able to be used w/ terraform
  tqdm.write(f"Number of .tf files that passed 'terraform init' and 'terraform validate': {len(valid_tf_file_paths)}/{len(terraform_files)} = {len(valid_tf_file_paths)/float(len(terraform_files)):.2f}")

  return valid_directory_path

# TODO: how would we actually validate that the terraform file does what the prompt intends?
# TODO: i'd (leon) think that considering we have the expected output, there's surely some properties we can use to compare truth vs. model response

if __name__ == "__main__":
  model_output_path = rf"{os.path.dirname(os.path.abspath(__file__))}/raw_outputs"
  print(validate_terraform_files(Path(model_output_path)))