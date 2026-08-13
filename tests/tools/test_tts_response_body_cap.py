import asyncio

import pytest

from tools import tts_tool


class Response:
    def __init__(self):
        self.closed = False

    async def aiter_bytes(self, chunk_size):
        del chunk_size
        for chunk in (b"12345", b"6789"):
            await asyncio.sleep(0)
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_streaming_response_is_bounded_and_closed():
    response = Response()
    with pytest.raises(RuntimeError, match="response exceeds 8 bytes"):
        await tts_tool._read_tts_response_bytes(
            response, label="TTS", limit=8
        )
    assert response.closed is True
