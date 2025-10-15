import pandas as pd
from pathlib import Path
import shutil
import click


@click.command()
@click.argument('path_to_dataset', type=click.Path(exists=True))
def copy_rtstructs(path_to_dataset:Path):
    path_to_dataset = Path(path_to_dataset)

    # setup the output directory to copy the RTSTRUCTs to
    destination_dir = path_to_dataset / "images" / "RTSTRUCTs"

    dataset_index_path = path_to_dataset / ".imgtools" / "images" / "index.csv"

    dataset_index = pd.read_csv(dataset_index_path)

    # Get just RTSTRUCT index data
    rtstruct_index = dataset_index[dataset_index["Modality"] == "RTSTRUCT"]

    print(rtstruct_index)

    for idx, rt in rtstruct_index.iterrows():
        rt_folder_path = Path(rt["folder"])
        dir_to_copy = path_to_dataset / rt_folder_path

        if dir_to_copy.exists() is False:
            print(f"WARNING: RTSTRUCT folder {dir_to_copy} does not exist, skipping...")
            continue

        else:
            print(f"Copying RTSTRUCT {rt['SeriesInstanceUID']}")
            # Skip the first part of the path since it's the original root, want to start from destination_dir
            dest_dir = destination_dir / Path(*rt_folder_path.parts[1:]) 

            copied_path = shutil.copytree(dir_to_copy, dest_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    copy_rtstructs()
