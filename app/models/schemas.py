from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    url: str
    transcript: str = ""  # optional — auto-fetched if empty


class AnalyzeMultiRequest(BaseModel):
    urls: list[str]  # multiple YouTube URLs to combine


class AnalyzeResponse(BaseModel):
    video_id: str
    duration_seconds: float
    frames: list[str]
    transcript: str
    description: str
    manin_prompt: str
    synthesized_title: str | None = None


class GenerateRequest(BaseModel):
    video_id: str
    transcript: str
    description: str
    duration: float
    title: str = "Inspired Video"
    orientation: str = "landscape"
    quality: str = "qm"
    voiceover: bool = True
    manin_prompt: str = ""  # editable manim prompt


class GenerateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    landscape_video: str | None = None
    portrait_video: str | None = None
    scene_file: str | None = None
    script_file: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
