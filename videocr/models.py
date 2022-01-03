from __future__ import annotations
from typing import List
from dataclasses import dataclass
from fuzzywuzzy import fuzz

@dataclass
class PredictedText:
    __slots__ = 'bounding_box', 'confidence', 'text'
    bounding_box: list
    confidence: float
    text: str


class PredictedFrames:
    start_index: int  # 0-based index of the frame
    end_index: int
    words: List[PredictedText]
    confidence: float  # total confidence of all words
    text: str

    def __init__(self, index: int, pred_data: list[list], conf_threshold: float):
        self.start_index = index
        self.end_index = index
        self.words = []

        total_conf = 0
        for l in pred_data:
            if len(l) < 2:
                continue
            bounding_box = l[0]
            text = l[1][0]
            conf = l[1][1]

            # word predictions with low confidence will be filtered out
            if conf >= conf_threshold:
                total_conf += conf
                self.words.append(PredictedText(bounding_box, conf, text))

        if self.words:
            self.confidence = total_conf/len(self.words)
            self.words.sort(key=lambda word: word.bounding_box[0][0])
        elif len(pred_data) == 0:
            self.confidence = 100
        else:
            self.confidence = 0
        self.text = ' '.join(word.text for word in self.words)

    def is_similar_to(self, other: PredictedFrames, threshold=70) -> bool:
        return fuzz.partial_ratio(self.text, other.text) >= threshold


class PredictedSubtitle:
    frames: List[PredictedFrames]
    sim_threshold: int
    text: str

    def __init__(self, frames: List[PredictedFrames], sim_threshold: int):
        self.frames = [f for f in frames if f.confidence > 0]
        self.frames.sort(key=lambda frame: frame.start_index)
        self.sim_threshold = sim_threshold

        if self.frames:
            self.text = max(self.frames, key=lambda f: f.confidence).text
        else:
            self.text = ''

    @property
    def index_start(self) -> int:
        if self.frames:
            return self.frames[0].start_index
        return 0

    @property
    def index_end(self) -> int:
        if self.frames:
            return self.frames[-1].end_index
        return 0

    def is_similar_to(self, other: PredictedSubtitle) -> bool:
        return fuzz.partial_ratio(self.text.replace(' ', ''), other.text.replace(' ', '')) >= self.sim_threshold

    def __repr__(self):
        return '{} - {}. {}'.format(self.index_start, self.index_end, self.text)
