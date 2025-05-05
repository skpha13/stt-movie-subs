from dataclasses import dataclass
from datetime import time
from typing import List


@dataclass
class Segment:
    start_time: time
    end_time: time
    subtitle: List[str]
