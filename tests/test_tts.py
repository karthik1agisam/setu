"""Tests for TTS interface (model-mocked)."""

from unittest.mock import MagicMock, patch


def test_synthesize_writes_wav(tmp_path):
    import torch

    from ai.speech.tts import TTS

    fake_tok = MagicMock()
    fake_tok.return_value = {"input_ids": [[1, 2]]}
    fake_mdl = MagicMock()
    fake_mdl.return_value.waveform = torch.tensor([[0.0] * 16000])

    with patch.object(TTS, "_load_mms", return_value=(fake_tok, fake_mdl)):
        out = TTS.synthesize("hello", "en", tmp_path / "x.wav")
    assert out.exists()
    data = out.read_bytes()
    assert data[:4] == b"RIFF"
    assert len(data) > 1000


def test_say_fallback(tmp_path):
    from ai.speech.tts import TTS

    with patch("ai.speech.tts.subprocess.run") as m:
        TTS.synthesize_say("hi", "en", tmp_path / "x.aiff")
    m.assert_called_once()
    assert m.call_args[0][0][0] == "say"
