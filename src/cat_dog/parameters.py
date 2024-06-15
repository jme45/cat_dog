"""'fix some parameters."""


from pathlib import Path
from torchvision.transforms import v2 as transf_v2

# Get the root dir of the package, so we can put the data into <Root>/data
# Need to take .parent 3 times, since taking it once only gets you to the
# directory the file is in and we need to go up 2 directories.
root_dir = Path(__file__).parent.parent.parent


DATA_ROOT_DIR = root_dir / "data"
DEFAULT_RUNS_DIR = root_dir / "runs"
DATA_AUGMENTATION_TRANSFORMS = transf_v2.TrivialAugmentWide()


BATCH_SIZE = 32
