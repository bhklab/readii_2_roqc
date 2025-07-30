from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from damply import dirs
from matplotlib.figure import Figure
from readii.io.loaders.general import loadImageDatasetConfig
from readii.io.writers.plot_writer import PlotWriter
from readii.utils import logger


def build_prediction_df(dataset_config: dict,
                        signature_name: str,
                        bootstrap_count: int = 1000,
                        evaluation_metric: str = "C-index",
                        ) -> pd.DataFrame:
    """Combine the bootstrapped prediction values for all image types into a single dataframe.
       Useful for plot functions like prediction_violin.

    Parameters
    ----------
    dataset_config
        Dataset configuration dictionary loaded from config/datasets directory.
    signature_name
        Name of signature used for making predictions. Used to set up data loading path.
    bootstrap_count
        Number of bootstrap iterations used to make prediction files. Used to set up data loading path.
    evaluation_metric
        Name of the evaluation metric used. Must match the column title in the prediction files in the bootstrap_count directory.
    
    Returns
    -------
    predictions : pd.DataFrame
        Dataframe where each column contains the evaluation metric values of each bootstrap prediction for an image type (e.g. the values in each file in the prediction directory)
    """
    dataset_name = dataset_config['DATASET_NAME']

    bootstrap_predictions_path = dirs.RESULTS / f"{dataset_config['DATA_SOURCE']}_{dataset_name}" / "prediction" / signature_name / f"bootstrap_{bootstrap_count}"

    predictions = pd.DataFrame()
    for image_predictions_file in sorted(bootstrap_predictions_path.rglob('*.csv')):
        image_type = str(image_predictions_file.stem)
        image_type_predictions = pd.read_csv(image_predictions_file, index_col=0)

        predictions[image_type] = image_type_predictions[evaluation_metric]

    return predictions



def prediction_violin(predictions: pd.DataFrame,
                      signature_name: str,
                      dataset_name: str | None = None,
                      title_text: str | None = None,
                      subtitle_text: str | None = None,
                      x_label: str = "Image Type",
                      y_label: str = "Concordance Index",
                      y_lower: float | None = 0.45,
                      y_upper: float | None = 0.85,
                      h_line: float | None = 0.5
                      ) -> Figure:
    """Generate a violin plot for each column in a given predictions dataframe.
    """

    # Initialize figure and axes
    fig, ax = plt.subplots()
    # Set width of figure
    fig.set_figwidth(9)
    # Rotate x-tick labels 90 degrees so they are readable and don't overlap with neighbours
    plt.xticks(rotation=90)

    # Create a plot with a violin for each column in the input dataframe
    ax = sns.violinplot(predictions)
    
    if h_line is not None:
        # If specified, draw a horizontal dashed black line at the given y-value 
        ax.axhline(y=h_line, 
                   color='black',
                   linestyle='--', 
                   linewidth=0.9)

    # Set y-axis boundary
    ax.set_ybound(y_lower, y_upper)

    # Set up title
    if title_text is None:
        title_text = f"OS Prediction for {signature_name} radiomic signature"
    plt.suptitle(title_text)

    # Set up subtitle
    if subtitle_text is None:
        subtitle_text = dataset_name
    ax.set_title(subtitle_text)

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig


def save_plot(plot_figure: Figure,
              plot_type: str,
              full_dataset_name: str,
              signature: str,
              data_description: str,
              evaluation_metric: str,
              overwrite: bool = False
              ) -> Path:
    plot_writer = PlotWriter(root_directory = dirs.RESULTS / full_dataset_name  /"visualization" / signature,
                            filename_format= "{PlotType}_{PlotDataDesc}_{Metric}.png",
                            overwrite=overwrite,
                            create_dirs=True)
    
    return plot_writer.save(plot_figure,
                            PlotType = plot_type,
                            PlotDataDesc = data_description,
                            Metric = evaluation_metric)


@click.command()
@click.argument('dataset', type=click.STRING)
@click.argument('signature', type=click.STRING)
@click.option('--overwrite', type=click.BOOL, default=False, help='Whether to overwrite existing plots. An error will be thrown if set to False and any plots exist.')
def plot(dataset: str,
         signature: str,
         overwrite: bool = False
         ) -> None:
    """Create and save out prediction plots for a given dataset and signature.
    Currently generates:
        * A violin plot with each image type

    Parameters
    ----------
    dataset: str
        Name of dataset to create plots for. Must have a configuration file in config/datasets.
    signature: str
        Name of the signature used to generate prediction results.
    overwrite: bool = False
        Whether to overwrite existing plot files.

    """
    # Input checking
    if dataset is None:
        message = "Dataset name must be provided."
        logger.error(message)
        raise ValueError(message)
    # Input checking
    if signature is None:
        message = "Signature name must be provided."
        logger.error(message)
        raise ValueError(message)
    
    # get path to dataset config directory
    config_dir_path = dirs.CONFIG / 'datasets'
    
    # Load in dataset configuration settings from provided dataset name
    dataset_config = loadImageDatasetConfig(dataset, config_dir_path)
    dataset_name = dataset_config['DATASET_NAME']
    full_dataset_name = f"{dataset_config['DATA_SOURCE']}_{dataset_config['DATASET_NAME']}"

    # Remove the file extension from the signature if it's included
    signature = signature.strip().removesuffix('.yaml')
    # Set up predictions dataframe for plotting
    predictions = build_prediction_df(dataset_config,
                                      signature_name = signature)

    # Make violin plot
    violin_fig = prediction_violin(predictions,
                                   signature_name = signature,
                                   dataset_name = dataset_name)
    
    # Save out the violin plot
    _output_path = save_plot(violin_fig,
                            plot_type = "violin",
                            full_dataset_name = full_dataset_name,
                            signature = signature,
                            data_description = "bootstrap_1000",
                            evaluation_metric = "C-index",
                            overwrite=overwrite
                            )



if __name__ == "__main__":
    plot()