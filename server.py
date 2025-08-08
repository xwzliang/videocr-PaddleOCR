import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from videocr import get_subtitles

# --- ensure logs directory exists ---
os.makedirs("logs", exist_ok=True)

# --- configure rotating file logger ---
log_path = "logs/server.log"
handler = RotatingFileHandler(
    log_path,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=1,  # keep one backup: server.log.1
)
handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
)
handler.setLevel(logging.INFO)

# root logger will catch everything
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

# also capture Uvicorn's logs
for uv_logger in ("uvicorn.error", "uvicorn.access"):
    lg = logging.getLogger(uv_logger)
    lg.addHandler(handler)

# --- FastAPI app ---
app = FastAPI()


class SubtitleRequest(BaseModel):
    file_path: str
    lang: str = "ch"
    time_start: str = "0:00"
    time_end: str = ""
    sim_threshold: int = 80
    conf_threshold: int = 50
    use_fullframe: bool = False
    use_gpu: bool = True
    brightness_threshold: int = 210
    similar_image_threshold: int = 1000
    similar_pixel_threshold: int = 25
    frames_to_skip: int = 1


@app.post("/subtitles")
async def subtitles(req: SubtitleRequest):
    logging.info(f"Request: {req.json()}")
    try:
        subs = get_subtitles(
            req.file_path,
            lang=req.lang,
            time_start=req.time_start,
            time_end=req.time_end,
            sim_threshold=req.sim_threshold,
            conf_threshold=req.conf_threshold,
            use_fullframe=req.use_fullframe,
            use_gpu=req.use_gpu,
            brightness_threshold=req.brightness_threshold,
            similar_image_threshold=req.similar_image_threshold,
            similar_pixel_threshold=req.similar_pixel_threshold,
            frames_to_skip=req.frames_to_skip,
        )
        outputs_dir = os.path.join(".", "outputs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(outputs_dir, f"{timestamp}.srt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(subs)
        logging.info("Subtitles generated successfully")
        return {"subtitles": subs}
    except Exception as e:
        logging.exception("Error in get_subtitles")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        # use our logging handler above
        access_log=True,
    )
