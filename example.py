from videocr import get_subtitles

if __name__ == "__main__":
    get_subtitles(
        "example_cropped.mp4",
        lang="ch",
        time_start="7:10",
        time_end="7:34",
        sim_threshold=80,
        conf_threshold=50,
        use_fullframe=True,
        use_gpu=True,
        brightness_threshold=210,
        similar_image_threshold=1000,
        similar_pixel_threshold=25,
        frames_to_skip=1,
    )
