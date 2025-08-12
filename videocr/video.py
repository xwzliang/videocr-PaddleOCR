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
        with Capture(path) as v:
            self.num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = v.get(cv2.CAP_PROP_FPS)
            self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        percent_crop_left: Optional[float] = None,
        percent_crop_right: Optional[float] = None,
        percent_keep_bottom: Optional[float] = None,
    ) -> None:
        # Initialize PaddleOCR
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

        # Seek to start timestamp
        if start_ms:
            ts = int((start_ms / 1000) / float(self.stream.time_base))
            self.container.seek(ts, any_frame=False, stream=self.stream)

        prev_gray = None
        skip_mod = frames_to_skip + 1
        frame_ct = 0
        self.pred_frames.clear()

        # Decode frames and run OCR
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
                h, w = img.shape[:2]

                # Crop based on percentages or absolute values
                if not use_fullframe:
                    if (
                        percent_crop_left is not None
                        or percent_crop_right is not None
                        or percent_keep_bottom is not None
                    ):
                        # compute pixel coords from percentages
                        left_px = int(w * (percent_crop_left or 0))
                        right_px = int(w * (percent_crop_right or 0))
                        bottom_h = int(
                            h
                            * (
                                percent_keep_bottom
                                if percent_keep_bottom is not None
                                else 1
                            )
                        )
                        # crop: keep bottom portion horizontally between left and right
                        img = img[h - bottom_h : h, left_px : w - right_px]
                    elif None not in (crop_x, crop_y, crop_width, crop_height):
                        img = img[
                            crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
                        ]
                    else:
                        # default: bottom third of frame
                        img = img[2 * h // 3 :, :]

                # Apply brightness mask
                if brightness_threshold:
                    mask = cv2.inRange(img, (brightness_threshold,) * 3, (255,) * 3)
                    img = cv2.bitwise_and(img, img, mask=mask)

                # Similarity filter
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

                # Perform OCR
                pf = PredictedFrames(frame_ct, self.ocr.ocr(img), conf_pct)
                pf.timestamp_ms = pts_ms
                self.pred_frames.append(pf)

    def get_subtitles(self, sim_threshold: int = 80) -> str:
        """
        Emit SRT cues by:
        • grouping same-text hits,
        • ending each cue at the next hit’s timestamp minus 1ms,
        • and for the last cue, adding one frame’s interval.
        """
        # 1) collect raw hits
        hits: List[tuple[float, str]] = []
        for pf in self.pred_frames:
            if not pf.lines:
                continue
            # flatten nested PredictedText lists
            pieces = [pt.text for line in pf.lines for pt in line]
            text = " ".join(pieces)
            hits.append((pf.timestamp_ms, text))

        if not hits:
            return ""

        # Precompute one frame interval in ms
        frame_interval_ms = 1000.0 / self.fps

        merge_gap = 3000.0  # max gap to still consider “same” run
        srt = []
        idx = 1
        i = 0
        n = len(hits)

        segments = []

        while i < n:
            start_ts, txt = hits[i]
            j = i + 1

            # extend run while text is similar and gap small
            while j < n:
                ts_j, txt_j = hits[j]
                ts_prev, _ = hits[j - 1]
                if (
                    fuzz.ratio(txt, txt_j) >= sim_threshold
                    and (ts_j - ts_prev) <= merge_gap
                ):
                    j += 1
                else:
                    break

            # determine end timestamp:
            if j < n:
                # there is a next, so end just before it
                end_ts = hits[j][0] - 1
            else:
                # last group → add one frame interval
                end_ts = hits[-1][0] + frame_interval_ms

            # format cue
            srt.append(
                f"{idx}\n"
                f"{self._fmt(start_ts)} --> {self._fmt(end_ts)}\n"
                f"{txt}\n\n"
            )
            segments.append(
                {
                    "start": start_ts,
                    "end": end_ts,
                    "text": txt,
                }
            )
            idx += 1
            i = j

        return "".join(srt), segments

    @staticmethod
    def _fmt(ms: float) -> str:
        total = int(ms)
        sec, msec = divmod(total, 1000)
        minute, sec = divmod(sec, 60)
        hour, minute = divmod(minute, 60)
        return f"{hour:02d}:{minute:02d}:{sec:02d},{msec:03d}"
