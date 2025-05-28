from damply import dirs as dmpdirs

rule run_mit_autopipeline:
    input:
        input_directory=dmpdirs.RAWDATA / "{DATA_SOURCE}_{DATASET_NAME}" / "images" / "{DATASET_NAME}",
        mit_crawl_index=dmpdirs.RAWDATA / "{DATA_SOURCE}_{DATASET_NAME}" / "images" / ".imgtools" / "{DATASET_NAME}"] / "index.csv"
    output:
        mit_autopipeline_index=dmpdirs.PROCDATA / "{DATA_SOURCE}_{DATASET_NAME}" / "images" / "mit_{DATASET_NAME}" / "mit_{DATASET_NAME}_index.csv",
        output_directory=directory(dmpdirs.PROCDATA / "{DATA_SOURCE}_{DATASET_NAME}" / "images" / "mit_{DATASET_NAME}")
    params:
        modalities=config["MIT"]["MODALITIES"],
        roi_match_map=config["MIT"]["ROI_MATCH_MAP"],
        roi_strategy=config["MIT"]["ROI_STRATEGY"]
    threads:
        4
    shell:
        """
        imgtools autopipeline {input.input_directory} {output.output_directory} \
        --modalities {params.modalities} \
        --roi-match-map {params.roi_match_map} \
        --roi-strategy {params.roi_strategy} \
        --jobs {threads} \
        --filename-format "{{PatientID}}_{{SampleNumber}}/{{Modality}}_{{SeriesInstanceUID}}/{{ImageID}}.nii.gz"
        """

rule run_mit_index:
    input:
        dicom_dir=dmpdirs.RAWDATA / "{DATA_SOURCE}_{DATASET_NAME}" / "images" / "{DATASET_NAME}",
    output:
        directory(dmpdirs.RAWDATA / "{DATA_SOURCE}_{DATASET_NAME}" / "images" / ".imgtools" / "{DATASET_NAME}",
        mit_crawl_index=dmpdirs.RAWDATA / "{DATA_SOURCE}_{DATASET_NAME}" / "images" / ".imgtools" / "{DATASET_NAME}" / "index.csv"
    threads:
        4
    shell:
        """
        imgtools index \
            --dicom-dir {input.dicom_dir} \
            --dataset-name {wildcards.DATASET_NAME} \
            --n-jobs {threads} \
            --force
        """