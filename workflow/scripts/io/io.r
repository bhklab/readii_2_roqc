#' Function to load in the feature data file for CPH model training or testing
#'
#' @param data_file_path A string path to the file to load.
#' 
#' @return A data.table containing the loaded data.
loadDataFile <- function(data_file_path) { #nolint
    checkmate::assertFile(data_file_path, access = "r", extension = c("csv", "xlsx"))

    switch(tools::file_ext(data_file_path),
        "csv" = read.csv(data_file_path, header = TRUE, sep = ",", check.names = FALSE),
        "xlsx" = readxl::read_excel(data_file_path)
    ) 
}


#' Function to load in a YAML file with proper checks
#'
#' @param yaml_file_path A string path to the file to load.
#' 
#' @return A data.table containing the loaded data.
loadYAMLFile <- function(yaml_file_path) { #nolint
    checkmate::assertFile(yaml_file_path, access = "r", extension = "yaml")
    yaml::read_yaml(yaml_file_path)
}


#' Function to read in a CPH signature file and get the feature names and weights
#' 
#' @param signature_name Name of the signature to read in, should have a signature.yaml file in the signatures folder. Weights are optional in the file.
#' 
#' @return list of feature names and weights
loadSignatureYAML <- function(signature_file_path) { #nolint 
    # Load the signature file
    signature_config <- loadYAMLFile(signature_file_path)
    # Names of the features in the signature
    sig_feature_names <- names(signature_config$signature)
    # Weights for the features in the signature
    sig_weights <- matrix(unlist(signature_config$signature))

    signature <- list(sig_feature_names, sig_weights)
    names(signature) <- c("names", "weights")

    return(signature)
}