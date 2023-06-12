__version__ = '10.0.10'

from pathlib import Path

from event_manager.yolo_tracking.boxmot.strongsort.strong_sort import StrongSORT
from event_manager.yolo_tracking.boxmot.ocsort.ocsort import OCSort as OCSORT
from event_manager.yolo_tracking.boxmot.bytetrack.byte_tracker import BYTETracker
from event_manager.yolo_tracking.boxmot.botsort.bot_sort import BoTSORT
from event_manager.yolo_tracking.boxmot.deepocsort.ocsort import OCSort as DeepOCSORT
from event_manager.yolo_tracking.boxmot.deep.reid_multibackend import ReIDDetectMultiBackend

from event_manager.yolo_tracking.boxmot.tracker_zoo import create_tracker, get_tracker_config


FILE = Path(__file__).resolve()
ROOT = FILE.parent  # root directory
EXAMPLES = ROOT / 'examples'
WEIGHTS = ROOT / 'weights'


__all__ = '__version__', 'StrongSORT', 'OCSORT', 'BYTETracker', 'BoTSORT',\
          'DeepOCSORT'  # allow simpler import
