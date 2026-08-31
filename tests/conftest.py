import sys
from pathlib import Path

# add the src directory to the sys.path to ensure that the prodml package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
