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


class PublishRequest(BaseModel):
    job_id: str
    title: str
    description: str
    tags: str
    platforms: list[str]  # ["youtube"]
    privacy: str = "public"
    video_type: str = "shorts"  # "shorts" or "regular"
    playlist: str = ""  # playlist name — created if doesn't exist


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
    publish: dict | None = None


class QueueVideoRequest(BaseModel):
    url: str
    source: str = "web"  # "web" | "extension" | "channel"
    channel_id: str | None = None
    source_short_youtube_id: str | None = None


class VideoUpdateRequest(BaseModel):
    title: str | None = None
    transcript: str | None = None
    description: str | None = None
    manin_prompt: str | None = None
    orientation: str | None = None
    quality: str | None = None
    voiceover: bool | None = None
    youtube_title: str | None = None
    youtube_description: str | None = None
    youtube_tags: str | None = None
    youtube_published_url: str | None = None
    youtube_published_id: str | None = None
    target_duration: float | None = None


class ChannelRequest(BaseModel):
    url: str
    name: str = ""
    notes: str = ""


class ChannelUpdateRequest(BaseModel):
    name: str | None = None
    notes: str | None = None
