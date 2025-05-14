source("workflow/scripts/io/io.r")
library(survcomp)

#' Function to test a CPH model with weights on a set of features.
#' 
#' @param test_labelled_features_file_path Dataframe containing the test features with outcome labels included.
#' @param surv_time_label Label for the time column in the test features file.
#' @param surv_event_label Label for the event column in the test features file.
#' @param model_feature_list List of feature names to use for the Cox model.
#' @param model_feature_weights Vector of weights for the Cox model
#' 
#' @return vector of test results.
testCoxModel <- function(labelled_feature_data,
                         surv_time_label,
                         surv_event_label,
                         model_feature_list,
                         model_feature_weights){ #nolint
    
    # Get only features selected for the model
    test_feature_data <- tryCatch({
        labelled_feature_data[, model_feature_list]
    }, error = function(e) {
        stop(paste("testCoxModel:Model features not found in provided feature set:", model_feature_list, "\n"))
    })

    # Convert the features dataframe to a matrix
    test_feature_matrix <- data.matrix(test_feature_data)

    # Multiply the feature matrix by the weights - this is applying the Cox model to the test set
    feature_hazards <- test_feature_matrix %*% model_feature_weights

    # Get the time and event label columns from the feature data
    time_label <- tryCatch({
        labelled_feature_data[, surv_time_label]
    }, error = function(e) {
        stop(paste("testCoxModel:Time column not found in provided feature set:", surv_time_label))
    })
    event_label <- tryCatch ({
        labelled_feature_data[, surv_event_label] 
    }, error = function(e) {
    stop(paste("testCoxModel:Event column not found in provided feature set:", surv_event_label))
    })

    # Calculate concordance index for the test set
    performance_results <- concordance.index(x = feature_hazards,
                                             surv.time = time_label,
                                             surv.event = event_label,
                                             method = "noether",
                                             alpha = 0.5,
                                             alternative = "two.sided")

    return(performance_results)
}


DATASET_NAME <- "TCIA_NSCLC-Radiomics"
DATA_DIR_PATH <- "data"
RAW_DATA_PATH <- paste0(DATA_DIR_PATH, "/rawdata/")
PROC_DATA_PATH <- paste0(DATA_DIR_PATH, "/procdata/", DATASET_NAME)
RESULTS_DATA_PATH <- paste0(DATA_DIR_PATH, "/results/", DATASET_NAME)


signature_name <- "aerts_original"
signature_file <- paste0(RAW_DATA_PATH, "/cph_weights_radiomic_signature/", signature_name, ".yaml")

image_type = "full_original"

feature_file <- paste0(PROC_DATA_PATH, "/signature_data/", signature_name, "/", image_type, ".csv")
feature_data <- loadDataFile(feature_file)

signature <- loadSignatureYAML(signature_file)
sig_feature_names <- signature$names
sig_weights <- signature$weights

results <- testCoxModel(labelled_feature_data=feature_data,
                         surv_time_label="survival_time_years",
                         surv_event_label="survival_event_binary",
                         model_feature_list=sig_feature_names,
                         model_feature_weights=sig_weights)