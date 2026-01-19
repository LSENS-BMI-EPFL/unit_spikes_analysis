import os
import shutil

# Base folders
axel_root = r"M:\analysis\Axel_Bisi\data"
myriam_root = r"M:\analysis\Myriam_Hamon\data"

# Loop only through MH* folders
for animal in os.listdir(axel_root):
    if not animal.startswith("MH"):
        continue  # skip AB* and other folders

    animal_path = os.path.join(axel_root, animal)

    if not os.path.isdir(animal_path):
        continue

    # Walk only inside MH* animal folders
    for root, dirs, files in os.walk(animal_path):
        if "ibl_format" in dirs:
            ibl_path = os.path.join(root, "ibl_format")

            # Compute relative path (keeps the MH* structure)
            rel_path = os.path.relpath(ibl_path, axel_root)
            myriam_path = os.path.join(myriam_root, rel_path)


            if not os.path.exists(myriam_path):
                print(f"Copying {ibl_path} -> {myriam_path}")
                shutil.copytree(ibl_path, myriam_path)
            else:  # If folder already exists, only put missing files
                for dirpath, _, filenames in os.walk(ibl_path):
                    rel_dir = os.path.relpath(dirpath, ibl_path)
                    dest_dir = os.path.join(myriam_path, rel_dir)

                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)

                    for f in filenames:
                        src_file = os.path.join(dirpath, f)
                        dest_file = os.path.join(dest_dir, f)

                        if not os.path.exists(dest_file):
                            shutil.copy2(src_file, dest_file)
                            print(f"Added missing file: {dest_file}")

print("✅ Copying finished.")
