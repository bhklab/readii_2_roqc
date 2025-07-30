# moved these out of the settings.py to make it cleaner
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ImageRegion(str, Enum):
	FULL = 'full'
	ROI = 'roi'
	NON_ROI = 'non-roi'


class ImagePermutation(str, Enum):
	ORIGINAL = 'original'
	SHUFFLE = 'shuffle'
	SAMPLE = 'sample'
	RANDOMIZE = 'randomize'


class CropBBoxConfig(BaseModel):
	min_dim_size: int = 4
	pad: int = 0


class CropCentroidConfig(BaseModel):
	# size is a tuple of (x, y, z)
	size: int = 50


class CropConfig(BaseModel):
	crop_bbox: CropBBoxConfig = Field(
		default_factory=CropBBoxConfig,
		description='Configuration for cropping images using bounding boxes.',
		title='Crop BBox Configuration',
	)

	crop_centroid: CropCentroidConfig = Field(
		default_factory=CropCentroidConfig,
		description='Configuration for cropping images using centroids.',
		title='Crop Centroid Configuration',
	)


class ImageTypes(BaseModel):
	regions: list[ImageRegion] = Field(
		default_factory=lambda: [
			ImageRegion.FULL,
			ImageRegion.ROI,
			ImageRegion.NON_ROI,
		],
		description='List of image regions to be processed.',
		title='Image Regions',
	)

	permutations: list[ImagePermutation] = Field(
		default_factory=lambda: [
			ImagePermutation.ORIGINAL,
			ImagePermutation.SHUFFLE,
			ImagePermutation.SAMPLE,
			ImagePermutation.RANDOMIZE,
		],
		description='List of image permutations to be applied.',
		title='Image Permutations',
	)

	crop: CropConfig = Field(
		default_factory=CropConfig,
		description='Configuration for cropping images, including bounding'
		' box and centroid cropping.',
		title='Crop Configuration',
	)


class TrainTestSplit(BaseModel):
	"""
	Configuration settings for train-test split.
	"""

	split: bool = Field(
		default=False,
		description='Whether to split the data.',
		title='Split Data',
	)

	split_variable: Dict[str, List[str]] = Field(
		default_factory=dict,
		description='What variable from CLINICAL.FILE to use to split the data and values to group by. '
		'Example: {"split_var": ["training", "test"]}',
		title='Split Variable',
	)

	impute: Optional[str] = Field(
		default=None,
		description='What to impute values in split_variable with. Should be one of the values provided in split_variable. '
		'If none provided, samples with no split value will be dropped.',
		title='Impute Value',
	)


class ReadiiSettings(BaseModel):
	"""
	Configuration settings for Readii.
	"""

	IMAGE_TYPES: ImageTypes = Field(
		default_factory=ImageTypes,
		description='Settings for image types, including regions and permutations.',
		title='Image Types',
	)

	TRAIN_TEST_SPLIT: TrainTestSplit = Field(
		default_factory=TrainTestSplit,
		description='Settings for train-test split.',
		title='Train Test Split',
	)
