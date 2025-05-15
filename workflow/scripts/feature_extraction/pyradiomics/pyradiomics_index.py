import pandas as pd
from pathlib import Path


def generate_dataset_index(image_directory:Path,
                           output_file_path:Path):
    # Construct file path lists for images and masks
    image_files = sorted(image_directory.rglob(pattern="*/CT*/CT.nii.gz"))
    mask_files = sorted(image_directory.rglob(pattern="*/RT*/GTV.nii.gz"))

    # Get list of sample IDs from top of data directory
    unique_sample_ids = [sample_dir.name for sample_dir in sorted(image_directory.glob(pattern="*/"))]

    if len(mask_files) > len(image_files):
        mask_index = pd.DataFrame(data= {'ID': [mask_path.parent.parent.stem for mask_path in mask_files],
                                         'Mask': mask_files})
        image_index = pd.DataFrame(data = {'ID': unique_sample_ids, 'Image': image_files})
        dataset_index = image_index.merge(mask_index, how='outer', left_on='ID', right_on='ID')

    else:
        # Construct dataframe to iterate over
        dataset_index = pd.DataFrame(data = {'ID': unique_sample_ids, 'Image': image_files, 'Mask': mask_files})

    dataset_index.to_csv(output_file_path, index=False)

    return

if __name__ == "__main__":
    # general data directory path setup
    DATA_DIR_PATH = Path("../../data")
    RAW_DATA_PATH = DATA_DIR_PATH / "rawdata"
    PROC_DATA_PATH = DATA_DIR_PATH / "procdata"
    RESULTS_DATA_PATH = DATA_DIR_PATH / "results"

    DATA_SOURCE = "TCIA"
    DATASET_NAME = "HEAD-NECK-RADIOMICS-HN1"

    dataset = f"{DATA_SOURCE}_{DATASET_NAME}"

    image_directory = RAW_DATA_PATH / dataset / "images"
    output_file_path = PROC_DATA_PATH / dataset / f"pyrad_{dataset}_index.csv"

    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    generate_dataset_index(image_directory, output_file_path)
    


