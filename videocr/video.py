from __future__ import annotations
from typing import List
import cv2
import av
import numpy as np
import os
from typing import Optional
from thefuzz import fuzz
from . import utils
from .models import PredictedFrames, PredictedSubtitle
from .opencv_adapter import Capture
from paddleocr import PaddleOCR


class Video:
    def __init__(
        self,
        path: str,
        det_model_dir: str,
        rec_model_dir: str,
    ):
        self.path = path
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir

        # open with PyAV
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

        # grab one frame to get height (for default crop)
        first = next(self.container.decode(video=0))
        self.height = first.height

        # rewind
        self.container.seek(0, any_frame=False, stream=self.stream)

        self.pred_frames: List[PredictedFrames] = []

    def _timecode_to_ms(self, tc: str) -> int:
        parts = list(map(float, tc.split(":")))
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            raise ValueError(f"Bad timecode '{tc}'")
        return int((h * 3600 + m * 60 + s) * 1000)

    def run_ocr(
        self,
        use_gpu: bool,
        lang: str,
        time_start: Optional[str],
        time_end: Optional[str],
        conf_threshold: int,
        use_fullframe: bool,
        brightness_threshold: int,
        similar_image_threshold: int,
        similar_pixel_threshold: int,
        frames_to_skip: int,
        crop_x: Optional[int],
        crop_y: Optional[int],
        crop_width: Optional[int],
        crop_height: Optional[int],
    ) -> None:
        # init OCR
        if utils.needs_conversion():
            self.ocr = PaddleOCR(
                lang=lang,
                text_recognition_model_dir=self.rec_model_dir,
                text_detection_model_dir=self.det_model_dir,
                text_detection_model_name=utils.get_model_name_from_dir(
                    self.det_model_dir
                ),
                text_recognition_model_name=utils.get_model_name_from_dir(
                    self.rec_model_dir
                ),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="gpu" if use_gpu else "cpu",
            )
        else:
            self.ocr = PaddleOCR(
                lang=lang,
                rec_model_dir=self.rec_model_dir,
                det_model_dir=self.det_model_dir,
                use_gpu=use_gpu,
            )

        start_ms = self._timecode_to_ms(time_start) if time_start else 0
        end_ms = self._timecode_to_ms(time_end) if time_end else None
        conf_pct = conf_threshold / 100.0

        # seek by PTS
        if start_ms:
            ts = int((start_ms / 1000) / float(self.stream.time_base))
            self.container.seek(ts, any_frame=False, stream=self.stream)

        prev_gray = None
        skip_mod = frames_to_skip + 1
        frame_ct = 0
        self.pred_frames.clear()

        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                pts_ms = float(frame.pts * self.stream.time_base) * 1000
                if pts_ms < start_ms:
                    continue
                if end_ms is not None and pts_ms > end_ms:
                    return

                frame_ct += 1
                if frame_ct % skip_mod != 0:
                    continue

                img = frame.to_ndarray(format="bgr24")

                # crop
                if not use_fullframe:
                    if None not in (crop_x, crop_y, crop_width, crop_height):
                        img = img[
                            crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
                        ]
                    else:
                        h = self.height
                        img = img[2 * h // 3 :, :]

                # brightness mask
                if brightness_threshold:
                    mask = cv2.inRange(img, (brightness_threshold,) * 3, (255,) * 3)
                    img = cv2.bitwise_and(img, img, mask=mask)

                # similarity filter
                if similar_image_threshold:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray)
                        _, diff = cv2.threshold(
                            diff, similar_pixel_threshold, 255, cv2.THRESH_BINARY
                        )
                        if np.count_nonzero(diff) < similar_image_threshold:
                            prev_gray = gray
                            continue
                    prev_gray = gray

                # OCR
                pf = PredictedFrames(frame_ct, self.ocr.ocr(img), conf_pct)
                pf.timestamp_ms = pts_ms
                self.pred_frames.append(pf)

    def get_subtitles(self, sim_threshold: int = 80) -> str:
        """
        Build SRT entries by:
         1) Extracting text from PredictedText objects.
         2) Merging hits that are similar and close in time.
        """
        # 1) collect (timestamp, text) hits
        hits: List[tuple[float, str]] = []
        for pf in self.pred_frames:
            if not getattr(pf, "lines", None):
                continue

            # extract each PredictedText.text
            pieces = []
            for line in pf.lines:
                for pt in line:
                    # PredictedText has attribute .text
                    pieces.append(pt.text if hasattr(pt, "text") else str(pt))
            text = " ".join(pieces)
            hits.append((pf.timestamp_ms, text))

        if not hits:
            return ""

        # 2) fuzzy-merge into SRT cues
        srt_entries = []
        idx = 1
        start_ts, last_ts, last_txt = hits[0][0], hits[0][0], hits[0][1]
        merge_gap = 3000.0  # 3 seconds max gap

        for ts, txt in hits[1:]:
            gap = ts - last_ts
            sim = fuzz.ratio(last_txt, txt)
            if sim >= sim_threshold and gap <= merge_gap:
                # extend current cue
                last_ts = ts
            else:
                # close out previous cue
                srt_entries.append(
                    f"{idx}\n"
                    f"{self._fmt(start_ts)} --> {self._fmt(last_ts+1)}\n"
                    f"{last_txt}\n"
                )
                idx += 1
                start_ts, last_ts, last_txt = ts, ts, txt

        # final cue
        srt_entries.append(
            f"{idx}\n"
            f"{self._fmt(start_ts)} --> {self._fmt(last_ts+1)}\n"
            f"{last_txt}\n"
        )

        return "".join(srt_entries)

    @staticmethod
    def _fmt(ms: float) -> str:
        total = int(ms)
        sec, msec = divmod(total, 1000)
        minute, sec = divmod(sec, 60)
        hour, minute = divmod(minute, 60)
        return f"{hour:02d}:{minute:02d}:{sec:02d},{msec:03d}"
