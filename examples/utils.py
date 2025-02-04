from typing import Optional, Union
import sys
from pathlib import Path


def get_aif360_location() -> Optional[str]:
    for d in sys.path:
        location = Path(d, 'aif360')
        if location.exists() and location.is_dir():
            return str(location)


def mv(files_to_move: str, destination: Union[str, Path]):
    dest_path = Path(destination)
    if not dest_path.exists() or not dest_path.is_dir():
        return
    if files_to_move.startswith('/'):
        top = '/'
        rest = files_to_move[1:]
    else:
        top = '.'
        rest = files_to_move
    for f in Path(top).glob(rest):
        Path(f).rename(Path(destination, Path(f).name))
