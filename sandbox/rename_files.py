import re
from pathlib import Path

proc_data_dir = Path("../data/procdata")
dataset_name = "NSCLC-Radiomics"
dataset_source = "TCIA"
roi_name = 'GTV'

path_to_files = proc_data_dir / (dataset_source + "_" + dataset_name) / 'pyradiomics_original_all_features'

for file in path_to_files.rglob('*.csv'):

    pattern = dataset_name + r"_\d{3}_"
    new_filename = re.sub(pattern, '', file.name)
    
    if not file.parent.name == roi_name:
        (file.parent / roi_name).mkdir(parents=True, exist_ok=True) 
        file.rename(file.parent / roi_name / new_filename)
