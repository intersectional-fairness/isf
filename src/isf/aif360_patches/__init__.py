from isf import aif360_patches
from pathlib import Path
import subprocess


def greetings():
    print("hello!")


def apply_patches():
    if len(aif360_patches.__path__) == 0:
        return
    holder_dir = aif360_patches.__path__[0]
    subprocess.run(str(Path(holder_dir, 'apply_patches.sh')))