import logging
import os
from pathlib import Path

logging.basicConfig(
	level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main() -> None:
	# examples of template defined env variables
	env_dir_variables = [
		'RAWDATA',
		'PROCDATA',
		'SCRIPTS',
	]

	for var in env_dir_variables:
		var_path = Path(os.environ.get(var))
		logger.info(f'{var}: {var_path} has {len(list(var_path.glob("*")))} files')


if __name__ == '__main__':
	logger.info(f'Starting example script from {Path().cwd()=}')
	main()
