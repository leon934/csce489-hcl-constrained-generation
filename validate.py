import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm


def strip_hcl_fence(raw_text: str) -> str:
  """Return Terraform content without surrounding ```hcl fences if present."""
  fence_pattern = re.compile(r"```(?:hcl)?\s*\n?(.*?)\s*```", re.IGNORECASE | re.DOTALL)
  match = fence_pattern.search(raw_text)
  if match:
    inner = match.group(1).strip()
    if inner and not inner.endswith("\n"):
      inner += "\n"
    return inner
  
  if raw_text.startswith("hcl```"):
    return raw_text.split("hcl```")[1]

  # no fences found; ensure newline termination for terraform fmt friendliness
  if raw_text and not raw_text.endswith("\n"):
    return raw_text + "\n"
  return raw_text

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
  total_files = len(terraform_files)

  if total_files == 0:
    return

  # first run terraform init
  # "terraform init" isn't required to be ran if there are no new files TO configure, but it's wtv if we run it anyways
  valid_tf_file_paths = []
  formatted_cache = {}
  fmt_pass_count = 0
  validate_pass_count = 0

  stripped_directory_path = Path(f"{path_to_terraform_files.parent}/stripped_outputs")
  if os.path.exists(stripped_directory_path):
    shutil.rmtree(stripped_directory_path)
  os.makedirs(stripped_directory_path, exist_ok=True)

  # creates a temporary directory to test each file individually
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir_path = Path(temp_dir)

    with tqdm(total=total_files, desc="Iterating through .tf files") as progress_bar:
      for tf_file_path in terraform_files:
        # remove the file from temp directory to test next one
        for temp_file in temp_dir_path.iterdir():
          if temp_file.is_file() or temp_file.is_symlink():
            temp_file.unlink()
          elif temp_file.is_dir():
            shutil.rmtree(temp_file)

        # copy the sanitized file into temp directory
        raw_text = tf_file_path.read_text(encoding="utf-8")
        sanitized_text = strip_hcl_fence(raw_text)
        formatted_cache[tf_file_path] = sanitized_text

        stripped_destination = stripped_directory_path / tf_file_path.name
        stripped_destination.write_text(sanitized_text, encoding="utf-8")

        destination_path = temp_dir_path / tf_file_path.name
        destination_path.write_text(sanitized_text, encoding="utf-8")

        def refresh_progress_bar():
          progress_bar.set_postfix(
            fmt=f"{fmt_pass_count}/{total_files} ({(fmt_pass_count/total_files)*100:.0f}%)",
            validate=f"{validate_pass_count}/{total_files} ({(validate_pass_count/total_files)*100:.0f}%)"
          )

        fmt_proc = subprocess.run(
          ["terraform", "fmt", tf_file_path.name],
          cwd=str(temp_dir_path),
          capture_output=True,
          text=True
        )

        if fmt_proc.returncode != 0:
          progress_bar.update(1)
          refresh_progress_bar()
          continue

        fmt_pass_count += 1

        formatted_text = destination_path.read_text(encoding="utf-8")
        formatted_cache[tf_file_path] = formatted_text
        stripped_destination.write_text(formatted_text, encoding="utf-8")

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
          progress_bar.update(1)
          refresh_progress_bar()
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
          validate_pass_count += 1

        progress_bar.update(1)
        refresh_progress_bar()

  valid_directory_path = Path(f"{path_to_terraform_files.parent}/valid_outputs")

  # deletes all existing files in valid_outputs directory from potentially previous iterations
  if os.path.exists(valid_directory_path):
    shutil.rmtree(valid_directory_path)
  
  os.makedirs(valid_directory_path, exist_ok=True)

  for terraform_file_path in valid_tf_file_paths:
    formatted_text = formatted_cache.get(terraform_file_path)
    if formatted_text is None:
      formatted_text = strip_hcl_fence(terraform_file_path.read_text(encoding="utf-8"))
    destination_path = valid_directory_path / terraform_file_path.name
    destination_path.write_text(formatted_text, encoding="utf-8")

  if terraform_files:
    fmt_ratio = fmt_pass_count / float(total_files)
    validate_ratio = validate_pass_count / float(total_files)
    tqdm.write(
      f"Number of .tf files that passed 'terraform fmt': {fmt_pass_count}/{len(terraform_files)} = {fmt_ratio:.2f}"
    )
    tqdm.write(
      f"Number of .tf files that passed 'terraform init' and 'terraform validate': {validate_pass_count}/{len(terraform_files)} = {validate_ratio:.2f}"
    )

  return valid_directory_path

# TODO: how would we actually validate that the terraform file does what the prompt intends?
# TODO: i'd (leon) think that considering we have the expected output, there's surely some properties we can use to compare truth vs. model response

if __name__ == "__main__":
  model_output_path = rf"{os.path.dirname(os.path.abspath(__file__))}/raw_outputs"
  valid_output_path = validate_terraform_files(Path(model_output_path))
  print(valid_output_path)
