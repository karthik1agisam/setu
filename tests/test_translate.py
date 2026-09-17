"""Tests for language detection + translator interface (model-mocked)."""

from unittest.mock import patch

from ai.translate.detect import detect_language


def test_detect_english():
    assert detect_language("Can I get PM-KISAN?") == "en"


def test_detect_hindi():
    assert detect_language("क्या मुझे पीएम-किसान मिलेगी?") == "hi"


def test_detect_telugu():
    assert detect_language("నాకు పీఎం-కిసాన్ వస్తుందా?") == "te"


def test_translate_same_language_passthrough():
    from ai.translate.indictrans import Translator

    assert Translator.translate("hello", "en", "en") == "hello"


def test_translate_routes_to_nllb_when_indictrans_unavailable():
    from ai.translate.indictrans import Translator

    with patch.object(Translator, "_use_indictrans", return_value=False), \
         patch.object(Translator, "_nllb", return_value="translated") as n:
        out = Translator.translate("నాకు స్కాలర్‌షిప్ వస్తుందా", "te", "en")
    assert out == "translated"
    n.assert_called_once_with("నాకు స్కాలర్‌షిప్ వస్తుందా", "te", "en")
