from readii_2_roqc.utils.loaders import load_dataset_config
from readii.io.writers.correlation_writer import CorrelationWriter
from damply import dirs

# More colour options
# from palettable.colorbrewer.diverging import PuOr_4




def self_correlate(dataset: str,
                   correlation_method:str,
                   extract_method:str,
                   extract_settings:str,
                   readii_permutation:str = "original",
                   readii_region:str = "full",
                   overwrite:bool = False
                   ):
    dataset_config, dataset_name, full_dataset_name = load_dataset_config(dataset)

    # Set up CorrelationWriter from readii
    corr_matrix_writer = CorrelationWriter(root_directory = dirs.RESULTS / full_dataset_name / "correlation" / "self",
                                           filename_format = extract_method / extract_settings / f"{readii_permutation}_{readii_region}_{correlation_method}.csv",
                                           overwrite = overwrite,
                                           create_dirs = True
    )
    # Option A: self-correlation only
    # need: 
    # - image type
    # - extraction method/config
    # - correlation method

    # Option B: cross-correlation (get self-correlation by default with this as well)
    # - image type vertical and horizontal


    return


if __name__ == "__main__":
    self_correlate()


    # 