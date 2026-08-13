import pytest

from tools import tts_tool


@pytest.mark.parametrize(
    "importer",
    [
        tts_tool._import_edge_tts,
        tts_tool._import_elevenlabs,
        tts_tool._import_mistral_client,
    ],
)
def test_optional_provider_imports_fail_directly_without_sync_installer(importer):
    try:
        importer()
    except ImportError:
        pass
