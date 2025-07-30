from __future__ import annotations

from pathlib import Path
from typing import Type

import yaml
from pydantic import Field
from pydantic_settings import (
	BaseSettings,
	PydanticBaseSettingsSource,
	SettingsConfigDict,
	# TomlConfigSettingsSource,
	YamlConfigSettingsSource,
)

from readii_2_roqc.mit_settings import MedImageToolsSettings
from readii_2_roqc.readii_settings import ReadiiSettings


class DatasetSettings(BaseSettings):
	"""
	    Central configuration class for R2R settings.

	    By default, it will look for a YAML file named `r2r-settings.yaml`
	    in the current working directory.

	    Otherwise, you can specify a different YAML file path
	when creating an instance of this class.

	    Example usage:
	    ```python
	    from readii_2_roqc.settings import DatasetSettings

	    settings = (
	        DatasetSettings()
	    )  # Loads from r2r-settings.yaml in the current directory
	    ```

	    You can also load settings from a different YAML file:
	    ```python
	    from readii_2_roqc.settings import DatasetSettings

	    settings = DatasetSettings.from_user_yaml(Path('path/to/your/settings.yaml'))
	    ```
	"""

	DATA_SOURCE: str = Field(
		description='Where you got the data from, i.e TCIA, MICCAI, PMCRT, etc.',
		title='Data Source',
	)

	DATASET_NAME: str = Field(
		description='Name of the dataset, e.g., NSCLC, LIDC-IDRI, etc.',
		title='Dataset Name',
	)

	MIT: MedImageToolsSettings = Field(
		description='Settings for MedImageTools, including image modalities and processing options.',
		title='MedImageTools Settings',
	)

	READII: ReadiiSettings = Field(
		description='Settings for Readii, including image types and permutations.',
		title='Readii Settings',
	)

	model_config = SettingsConfigDict(
		# commented this out to avoid confusion
		# default path to the YAML settings file
		# if you want to use a different file, you can pass it as an argument
		# to the DatasetSettings constructor or use the from_user_yaml method
		# yaml_file=(Path().cwd() / 'r2r-settings.yaml',),
		# extra allow for other fields to be present in the config file
		# this allows for the config file to be used for other purposes
		# but also for users to define anything else they might want
		extra='ignore',
	)

	@classmethod
	def from_user_yaml(cls, path: Path) -> DatasetSettings:
		"""Load settings from a YAML file."""
		source = YamlConfigSettingsSource(cls, yaml_file=path)
		settings = source()
		return cls(**settings)

	@classmethod
	def settings_customise_sources(
		cls,
		settings_cls: Type[BaseSettings],
		init_settings: PydanticBaseSettingsSource,
		env_settings: PydanticBaseSettingsSource,
		dotenv_settings: PydanticBaseSettingsSource,
		file_secret_settings: PydanticBaseSettingsSource,
	) -> tuple[PydanticBaseSettingsSource, ...]:
		# this signifies that the settings will be loaded from
		return (
			init_settings,
			YamlConfigSettingsSource(settings_cls),
		)

	@property
	def COMBINED_DATA_NAME(self) -> str:  # noqa: N802
		"""Return the combined data name based on the data source."""
		return f'{self.DATA_SOURCE}_{self.DATASET_NAME}'

	@property
	def dicom_dir(self) -> Path:
		"""Return the expected path to the DICOM directory."""
		return Path(self.COMBINED_DATA_NAME) / 'images' / self.DATASET_NAME

	@property
	def mit_crawl_index(self) -> Path:
		"""Return the expected path to the crawl index file for MedImageTools."""
		return (
			Path(self.COMBINED_DATA_NAME)
			/ 'images'
			/ '.imgtools'
			/ self.DATASET_NAME
			/ 'index.csv'
		)

	@property
	def mit_autopipeline_simple_index(self) -> Path:
		"""Return the expected path to the simple index file for MedImageTools."""
		return (
			Path(self.COMBINED_DATA_NAME)
			/ 'images'
			/ f'mit_{self.DATASET_NAME}'
			/ f'mit_{self.DATASET_NAME}_index-simple.csv'
		)

	@property
	def mit_autopipeline_index(self) -> Path:
		"""Return the expected path to the autopipeline index file for MedImageTools."""
		return (
			Path(self.COMBINED_DATA_NAME)
			/ 'images'
			/ f'mit_{self.DATASET_NAME}'
			/ f'mit_{self.DATASET_NAME}_index.csv'
		)

	@property
	def json_schema(self) -> dict:
		"""Return the JSON schema for the settings."""
		return self.model_json_schema()

	def to_yaml(self, path: Path, indent: int = 4) -> None:
		"""Return the YAML representation of the settings."""
		path = Path(path).resolve()
		model = self.model_dump(mode='json')
		try:
			path.parent.mkdir(parents=True, exist_ok=True)
			with path.open('w') as f:
				yaml.dump(model, f, sort_keys=False, indent=indent)
		except (OSError, IOError) as e:
			msg = f'Failed to save settings to {path}: {e}'
			raise ValueError(msg) from e

	@classmethod
	def default(cls) -> DatasetSettings:
		"""Return a default instance of DatasetSettings."""
		from imgtools.coretypes import ROIMatcher  # noqa

		return DatasetSettings(
			DATA_SOURCE='DefaultDataSource',
			DATASET_NAME='DefaultDatasetName',
			MIT=MedImageToolsSettings(
				ROI_MATCHER=ROIMatcher(match_map={'ROI': ['.*']})
			),
			READII=ReadiiSettings(),
		)


if __name__ == '__main__':
	from rich.console import Console

	console = Console()

	# this will create default settings
	config = DatasetSettings.default()
	console.print(config)

	# this will load from a yaml file

	config = DatasetSettings.from_user_yaml(
		Path.cwd() / 'config' / 'datasets' / 'settings' / 'RADCURE.yaml'
	)
	console.print(config)

	# console.print(f'Settings saved to r2r-settings.yaml')
