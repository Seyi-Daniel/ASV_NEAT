from __future__ import annotations
import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state","action","reward","next_state","done"))

class Replay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, bs: int) -> Transition:
        batch = random.sample(self.buf, bs)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buf)