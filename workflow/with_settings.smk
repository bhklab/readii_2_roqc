from damply import dirs as dmpdirs
from readii_2_roqc import DatasetSettings

# Option 1: specify the config file directly
# configfile: dmpdirs.CONFIG / "datasets" / "settings" / "r2r-settings.yaml"

# Option 2: specify the config file via command line argument
# `--configfile $CONFIG/datasets/settings/r2r-settings.yaml`

settings = DatasetSettings(**config)

print(f"Processing dataset: {settings.COMBINED_DATA_NAME}")

rule all:
    # Target rule
    input:
        mit_niftis_directory=dmpdirs.PROCDATA / settings.COMBINED_DATA_NAME / "images" / f"mit_{settings.DATASET_NAME}",
    params:
        settings=settings
    script:
        # just a script to show how to use the settings in a python script
        dmpdirs.SCRIPTS / "mit" / "all.py"


rule run_mit_autopipeline:
    input:
        input_directory=dmpdirs.RAWDATA / settings.COMBINED_DATA_NAME / "images" / settings.DATASET_NAME,
        mit_crawl_index=dmpdirs.RAWDATA / settings.mit_crawl_index
    output:
        mit_autopipeline_index = dmpdirs.PROCDATA / settings.mit_autopipeline_index,
        mit_autopipeline_simple_index = dmpdirs.PROCDATA / settings.mit_autopipeline_simple_index,
        output_directory=directory(dmpdirs.PROCDATA / settings.COMBINED_DATA_NAME / "images" / f"mit_{settings.DATASET_NAME}")
    threads:
        2
    shell:
        """
        imgtools autopipeline {input.input_directory} {output.output_directory} \
            --modalities {settings.MIT.modalities_str} \
            {settings.MIT.mit_rmap_str} \
            --roi-strategy {settings.MIT.roi_strategy} \
            --jobs {threads} \
            --filename-format "{{PatientID}}_{{SampleNumber}}/{{Modality}}_{{SeriesInstanceUID}}/{{ImageID}}.nii.gz"
        """

rule run_mit_index:
    input:
        dicom_dir=dmpdirs.RAWDATA / settings.dicom_dir
    output:
        mit_crawl_index=dmpdirs.RAWDATA / settings.mit_crawl_index
    threads:
        2
    shell:
        """
        imgtools index \
            --dicom-dir {input.dicom_dir} \
            --dataset-name {settings.DATASET_NAME} \
            --force \
            --n-jobs {threads}
        """