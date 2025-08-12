from typing import Optional
from .video import Video


def get_subtitles(
    video_path: str,
    lang="ch",
    time_start="0:00",
    time_end="",
    conf_threshold=75,
    sim_threshold=80,
    use_fullframe=False,
    det_model_dir=None,
    rec_model_dir=None,
    use_gpu=False,
    brightness_threshold=None,
    similar_image_threshold=100,
    similar_pixel_threshold=25,
    frames_to_skip=1,
    crop_x=None,
    crop_y=None,
    crop_width=None,
    crop_height=None,
    percent_crop_left: Optional[float] = None,
    percent_crop_right: Optional[float] = None,
    percent_keep_bottom: Optional[float] = None,
):

    v = Video(video_path, det_model_dir, rec_model_dir)
    v.run_ocr(
        use_gpu,
        lang,
        time_start,
        time_end,
        conf_threshold,
        use_fullframe,
        brightness_threshold,
        similar_image_threshold,
        similar_pixel_threshold,
        frames_to_skip,
        crop_x,
        crop_y,
        crop_width,
        crop_height,
        percent_crop_left=percent_crop_left,
        percent_crop_right=percent_crop_right,
        percent_keep_bottom=percent_keep_bottom,
    )
    return v.get_subtitles(sim_threshold)


def save_subtitles_to_file(
    video_path: str,
    file_path="subtitle.srt",
    lang="ch",
    time_start="0:00",
    time_end="",
    conf_threshold=75,
    sim_threshold=80,
    use_fullframe=False,
    det_model_dir=None,
    rec_model_dir=None,
    use_gpu=False,
    brightness_threshold=None,
    similar_image_threshold=100,
    similar_pixel_threshold=25,
    frames_to_skip=1,
    crop_x=None,
    crop_y=None,
    crop_width=None,
    crop_height=None,
    percent_crop_left: Optional[float] = None,
    percent_crop_right: Optional[float] = None,
    percent_keep_bottom: Optional[float] = None,
) -> None:
    with open(file_path, "w+", encoding="utf-8") as f:
        f.write(
            get_subtitles(
                video_path,
                lang,
                time_start,
                time_end,
                conf_threshold,
                sim_threshold,
                use_fullframe,
                det_model_dir,
                rec_model_dir,
                use_gpu,
                brightness_threshold,
                similar_image_threshold,
                similar_pixel_threshold,
                frames_to_skip,
                crop_x,
                crop_y,
                crop_width,
                crop_height,
                percent_crop_left=percent_crop_left,
                percent_crop_right=percent_crop_right,
                percent_keep_bottom=percent_keep_bottom,
            )[0]
        )
