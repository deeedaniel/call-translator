from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from config import settings

router = APIRouter(prefix="/twiml", tags=["twiml"])

TWIML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/ws/twilio" />
  </Connect>
  <Say>Please hold while we connect your interpreter.</Say>
</Response>"""


@router.post("/voice")
async def voice_webhook(request: Request) -> Response:
    """Return TwiML instructing Twilio to open a Media Stream WebSocket."""
    host = settings.public_host or request.headers.get("host", "localhost:8000")
    xml = TWIML_TEMPLATE.format(host=host)
    return Response(content=xml, media_type="application/xml")
