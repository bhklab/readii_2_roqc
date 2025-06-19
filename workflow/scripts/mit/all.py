from rich.console import Console

console = Console()


configfile = snakemake

# Option 1
# pass settings into rule params, and then load them from snakemake.params
# they will already be of type DatasetSettings
console.print(snakemake.params.settings)


# Option 2
# load settings from the automatically passed in snakemake config (read in from --configfile)
from readii_2_roqc import DatasetSettings

settings2 = DatasetSettings(**snakemake.config)
# console.print(settings2)


assert snakemake.params.settings == settings2
