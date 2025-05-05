from pathlib import Path
import re

proc_data_dir = Path("../data/procdata")
dataset_name = "HEAD-NECK-RADIOMICS-HN1"
dataset_source = "TCIA"

path_to_files = proc_data_dir / (dataset_source + "_" + dataset_name) / 'pyradiomics_original_all_features'

for file in path_to_files.rglob('*.csv'):

    pattern = dataset_name + r"_\d{3}_"
    new_filename = re.sub(pattern, '', file.name)
    
    if not file.parent.name == 'GTV-1':
        (file.parent / "GTV-1").mkdir(parents=True, exist_ok=True) 
        file.rename(file.parent / "GTV-1" / new_filename)
