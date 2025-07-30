# moved these out of the settings.py to make it cleaner
from __future__ import annotations

from enum import Enum

from imgtools.coretypes import ROIMatcher
from pydantic import BaseModel, Field


class ImageModality(str, Enum):
	"""Enum for image modalities.
	Defines the valid modalities that can be used in the MedImageToolsSettings.
	"""

	CT = 'CT'
	MR = 'MR'


class MaskModality(str, Enum):
	"""Enum for mask modalities.
	Defines the valid modalities that can be used in the MedImageToolsSettings.
	"""

	RTSTRUCT = 'RTSTRUCT'
	SEG = 'SEG'


class ModalityPair(BaseModel):
	image: ImageModality = Field(
		default=ImageModality.CT,
		description='Image modality to be processed.',
		title='Image Modality',
	)
	mask: MaskModality = Field(
		default=MaskModality.RTSTRUCT,
		description='Mask modality to be processed.',
		title='Mask Modality',
	)

	def __str__(self) -> str:
		"""Return a string representation in the format 'CT,RTSTRUCT'."""
		return f'{self.image.value},{self.mask.value}'

	def to_str(self) -> str:
		"""Return a comma-separated string of the modality values."""
		return str(self)


class MedImageToolsSettings(BaseModel):
	"""
	Configuration settings for MedImageTools.
	Defines how images are processed, including region handling and image permutations.
	"""

	MODALITIES: ModalityPair = Field(
		default_factory=ModalityPair,
		description='Configuration for image and mask modalities to be processed.',
		title='Image Modalities',
	)

	ROI_MATCHER: ROIMatcher = Field(
		# default_factory=lambda: ROIMatcher(match_map={'ROI': ['.*']}),
		description='Configuration for ROI (Region of Interest) matching in segmentation data.'
		' Defines how ROIs are identified, matched and processed from RTSTRUCT or SEG files.',
		title='ROI Matcher Configuration',
	)

	# uncomment this if you want to make sure theres no extra fields under MIT
	# model_config = {'extra': 'forbid'}

	@property
	def modalities_str(self) -> str:
		"""Return a string representation of the modalities."""
		return str(self.MODALITIES)

	@property
	def mit_rmap_str(self) -> str:
		"""Helper to generate the autopipeline parameter for -rmap"""
		match_map: dict[str, list[str]] = self.ROI_MATCHER.match_map
		return ' '.join(
			[f'--roi-match-map={k}:{",".join(v)}' for k, v in match_map.items()]
		)

	@property
	def roi_strategy(self) -> str:
		"""Return the ROI strategy as a string."""
		return self.ROI_MATCHER.handling_strategy.value.upper()


if __name__ == '__main__':
	# example usage
	from rich.console import Console

	console = Console()

	settings = MedImageToolsSettings(
		modalities=ModalityPair(image=ImageModality.CT, mask=MaskModality.SEG),
		roi_matcher=ROIMatcher(match_map={'ROI': ['.*']}),
	)
	console.print_json(settings.model_dump_json(indent=2))
	console.print(settings.roi_strategy)
