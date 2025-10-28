import pandas as pd
import SimpleITK as sitk

from damply import dirs
from pathlib import Path
from tqdm import tqdm

from imgtools.autopipeline import SIMPLIFIED_COLUMNS
from imgtools.transforms.functional import resize
from imgtools.io.writers.nifti_writer import NIFTIWriter, NiftiWriterIOError
from readii.utils import logger

def resize_to_image(mask:sitk.Image, 
                    image:sitk.Image,
                    interpolation:str | None = 'nearest'
                    ) -> sitk.Image:
    """Resize and resample the mask with interpolation (nearest neighbour by default) to the dimensions of the image.
    """
    if interpolation is None:
        interpolation = 'nearest'

    resized_mask = resize(mask, 
                          size = list(image.GetSize()), 
                          interpolation = interpolation)
    
    # This can end up being returned as a float32 image, so need to cast to uint8 to avoid issues with label values
    resized_mask = sitk.Cast(resized_mask, sitk.sitkUInt8)

    return resized_mask


def process_nifti_mask(mask_path:Path,
                       image_path:Path,
                       interpolation:str | None = None,
                       ) -> sitk.Image:
    
    # load mask 
    mask = sitk.ReadImage(mask_path)

    # load image
    image = sitk.ReadImage(image_path)

    # resize the mask to the image
    return resize_to_image(mask, image, interpolation=interpolation)


def update_simplified_index(index_path:Path) -> pd.DataFrame:
    # Write simplified index file
    simple_index = index_path.parent / f"{index_path.stem}-simple.csv"

    try:
        index_df = pd.read_csv(index_path)

        # Get columns in the order we want
        # If a column is not in the index_df, it will be filled with NaN
        simple_index_df = index_df.reindex(columns=SIMPLIFIED_COLUMNS)

        # Sort by 'filepath' to make it easier to read
        if "filepath" in simple_index_df.columns:
            simple_index_df = simple_index_df.sort_values(by=["filepath"])

        simple_index_df.to_csv(simple_index, index=False)
        logger.info(f"Index file saved to {simple_index}")
    except Exception as e:
        logger.error(f"Failed to create simplified index: {e}")
    
    return simple_index_df


if __name__ == '__main__':
    dataset_name = 'TCGA-KIRC'
    full_data_name = 'TCIA_TCGA-KIRC'

    mask_index_file = dirs.RAWDATA / full_data_name / dataset_name / "list.csv"
    mask_index = pd.read_csv(mask_index_file, index_col=0)

    proc_images_dir = dirs.PROCDATA / full_data_name / "images" / f"mit_{dataset_name}"
    index_path = proc_images_dir / f"mit_{dataset_name}_index.csv"

    nifti_writer = NIFTIWriter(
            root_directory = proc_images_dir,
            filename_format = "{PatientID}_{SampleNumber}/{Modality}_{SeriesInstanceUID}/{ImageID}.nii.gz",
            create_dirs = True,
            existing_file_mode = 'OVERWRITE',
            sanitize_filenames = True,
            index_filename = index_path,
            overwrite_index = False
        )

    for _idx, mask_data in tqdm(mask_index.iterrows(), desc="Processing NIFTIs like MIT...", total=len(mask_index)):
        # Use details from the mask index to find the base image nifti file processed by MIT
        # need the last 8 digists of the SeriesInstanceUID
        series_inst_uid_end = mask_data.SeriesInstanceUID[-8:]
        # Construct the search pattern using the patientID and SeriesInstanceUID
        search_pattern = f"{mask_data.Patient}*/*_{series_inst_uid_end}/*.nii.gz"

        logger.info("Searching for base image for mask...")
        # Search for a nifti file in the MIT directory structure using the search pattern
        try:
            image_path = proc_images_dir.glob(search_pattern).__next__()
        except StopIteration:
            print(f"No image found for {mask_data.Patient} with SeriesInstanceUID {series_inst_uid_end}. Skipping.")
            continue

        sample_id = image_path.parent.parent.name[-4:]

        mask_path = mask_index_file.parent / mask_data['mask']
        # get the number of roi for this segmentation (some patients have multiple rois)
        # THIS IS TCGA-KIRC specific
        roi_num = mask_path.name.removesuffix('_.nii.gz')[-1]

        logger.info(f'Processing {mask_data.Patient} segmentation {roi_num}...')

        processed_mask = process_nifti_mask(mask_path,
                                            image_path,
                                            interpolation = 'nearest'
                                            )
        
        logger.info('Saving out the processed mask...')
        out_path = nifti_writer.save(processed_mask,
                                     PatientID = mask_data['Patient'],
                                     SampleNumber = sample_id,
                                     SeriesInstanceUID = f"{mask_data['SeriesInstanceUID']}_{roi_num}",
                                     ReferencedSeriesUID = mask_data['SeriesInstanceUID'],
                                     ImageID = f"ROI__[GTV_{roi_num}]",
                                     Modality = 'SEG',
                                     roi_key = 'ROI',
                                     matched_rois = f"GTV_{roi_num}",
                                     dtype_str = "8-bit unsigned integer",
                                     ndim = 3,
                                     )

        logger.info(f'Mask saved to {out_path}')


    simple_index = update_simplified_index(index_path)