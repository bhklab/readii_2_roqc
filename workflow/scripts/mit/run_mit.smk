from damply import dirs as dmpdirs

rule run_mit_autopipeline:
    input:
        input_directory=dmpdirs.RAWDATA / COMBINED_DATA_NAME / "images",
        mit_crawl_index=dmpdirs.RAWDATA / COMBINED_DATA_NAME / ".imgtools" / config["DATASET_NAME"] / "index.csv"
    output:
        mit_autopipeline_index=dmpdirs.PROCDATA / COMBINED_DATA_NAME / "images" / f"mit_{config["DATASET_NAME"]}" / f"mit_{config["DATASET_NAME"]}_index.csv",
        mit_simple_index_file=dmpdirs.PROCDATA / COMBINED_DATA_NAME / "images" / f"mit_{config["DATASET_NAME"]}" / f"mit_{config["DATASET_NAME"]}_index-simple.csv",
        output_directory=directory(dmpdirs.PROCDATA / COMBINED_DATA_NAME / "images" / f"mit_{config["DATASET_NAME"]}")
    params:
        modalities=f"{config["MIT"]["MODALITIES"]["image"]},{config["MIT"]["MODALITIES"]["mask"]}",
        roi_match_map=config["MIT"]["ROI_MATCH_MAP"],
        roi_strategy=config["MIT"]["ROI_STRATEGY"]
    shell:
        """
        imgtools autopipeline {input.input_directory} {output.output_directory} \
        --modalities {params.modalities} \
        --roi-match-map {params.roi_match_map} \
        --roi-strategy {params.roi_strategy} \
        --filename-format "{{PatientID}}_{{SampleNumber}}/{{Modality}}_{{SeriesInstanceUID}}/{{ImageID}}.nii.gz"
        """

rule run_mit_index:
    input:
        dicom_dir=dmpdirs.RAWDATA / COMBINED_DATA_NAME / "images"
    output:
        directory(dmpdirs.RAWDATA / COMBINED_DATA_NAME / ".imgtools" / config["DATASET_NAME"]),
        mit_crawl_index=dmpdirs.RAWDATA / COMBINED_DATA_NAME / ".imgtools" / config["DATASET_NAME"] / "index.csv"
    params:
        dataset_name= config["DATASET_NAME"]
    shell:
        """
        imgtools index --dicom-dir {input.dicom_dir} --dataset-name {params.dataset_name} --force
        """