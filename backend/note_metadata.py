"""
Nepali Currency Note Denomination Metadata.
Rich metadata with bilingual voice templates for TTS output.
Secondary advanced module of the Devanagari Character Recognition project.
"""

NOTE_METADATA = {
    # ── Unknown / not-recognized sentinel ─────────────────────────
    "unknown": {
        "value": 0,
        "label": "unknown",
        "english_name": "Not Recognized",
        "nepali_name": "पहिचान गर्न सकिएन",
        "nepali_numeral": "–",
        "color_hint": "–",
        "voice": {
            "english": "This note could not be recognized. Please try with a clearer, well-lit image of a Nepali banknote.",
            "nepali": "यो नोट पहिचान गर्न सकिएन। कृपया स्पष्ट र राम्रोसँग प्रकाशित नेपाली नोटको तस्बिर प्रयोग गर्नुहोस्।",
            "mixed": "Note not recognized. पहिचान गर्न सकिएन। Please try again.",
            "confidence_high": "This image was not recognized as a Nepali currency note.",
            "confidence_low": "Confidence is too low to identify this note. Please try with a clearer image.",
            "confidence_high_nepali": "यो नेपाली नोट भनेर पहिचान हुन सकेन।",
            "confidence_low_nepali": "विश्वास धेरै कम छ। कृपया स्पष्ट तस्बिर प्रयोग गर्नुहोस्।",
        },
    },

    # ── Denominations ──────────────────────────────────────────────
    "rs5": {
        "value": 5,
        "label": "rs5",
        "english_name": "Five Rupees",
        "nepali_name": "पाँच रुपैयाँ",
        "nepali_numeral": "५",
        "color_hint": "brown/red",
        "voice": {
            "english": "This is a five rupee note. Denomination: five Nepali rupees.",
            "nepali": "यो पाँच रुपैयाँको नोट हो। मूल्य: पाँच नेपाली रुपैयाँ।",
            "mixed": "Detected denomination: 5 rupees. नेपालीमा: पाँच रुपैयाँको नोट।",
            "confidence_high": "Recognized value is 5 rupees with high confidence.",
            "confidence_low": "This might be a 5 rupee note, but confidence is low. Please try a clearer image.",
            "confidence_high_nepali": "पाँच रुपैयाँको नोट भनेर उच्च विश्वासका साथ पहिचान भयो।",
            "confidence_low_nepali": "यो पाँच रुपैयाँको नोट हुनसक्छ, तर विश्वास कम छ।",
        },
    },
    "rs10": {
        "value": 10,
        "label": "rs10",
        "english_name": "Ten Rupees",
        "nepali_name": "दश रुपैयाँ",
        "nepali_numeral": "१०",
        "color_hint": "brown/orange",
        "voice": {
            "english": "This is a ten rupee note. Denomination: ten Nepali rupees.",
            "nepali": "यो दश रुपैयाँको नोट हो। मूल्य: दश नेपाली रुपैयाँ।",
            "mixed": "Detected denomination: 10 rupees. नेपालीमा: दश रुपैयाँको नोट।",
            "confidence_high": "Recognized value is 10 rupees with high confidence.",
            "confidence_low": "This might be a 10 rupee note, but confidence is low. Please try a clearer image.",
            "confidence_high_nepali": "दश रुपैयाँको नोट भनेर उच्च विश्वासका साथ पहिचान भयो।",
            "confidence_low_nepali": "यो दश रुपैयाँको नोट हुनसक्छ, तर विश्वास कम छ।",
        },
    },
    "rs20": {
        "value": 20,
        "label": "rs20",
        "english_name": "Twenty Rupees",
        "nepali_name": "बीस रुपैयाँ",
        "nepali_numeral": "२०",
        "color_hint": "orange",
        "voice": {
            "english": "This is a twenty rupee note. Denomination: twenty Nepali rupees.",
            "nepali": "यो बीस रुपैयाँको नोट हो। मूल्य: बीस नेपाली रुपैयाँ।",
            "mixed": "Detected denomination: 20 rupees. नेपालीमा: बीस रुपैयाँको नोट।",
            "confidence_high": "Recognized value is 20 rupees with high confidence.",
            "confidence_low": "This might be a 20 rupee note, but confidence is low. Please try a clearer image.",
            "confidence_high_nepali": "बीस रुपैयाँको नोट भनेर उच्च विश्वासका साथ पहिचान भयो।",
            "confidence_low_nepali": "यो बीस रुपैयाँको नोट हुनसक्छ, तर विश्वास कम छ।",
        },
    },
    "rs50": {
        "value": 50,
        "label": "rs50",
        "english_name": "Fifty Rupees",
        "nepali_name": "पचास रुपैयाँ",
        "nepali_numeral": "५०",
        "color_hint": "blue/purple",
        "voice": {
            "english": "This is a fifty rupee note. Denomination: fifty Nepali rupees.",
            "nepali": "यो पचास रुपैयाँको नोट हो। मूल्य: पचास नेपाली रुपैयाँ।",
            "mixed": "Detected denomination: 50 rupees. नेपालीमा: पचास रुपैयाँको नोट।",
            "confidence_high": "Recognized value is 50 rupees with high confidence.",
            "confidence_low": "This might be a 50 rupee note, but confidence is low. Please try a clearer image.",
            "confidence_high_nepali": "पचास रुपैयाँको नोट भनेर उच्च विश्वासका साथ पहिचान भयो।",
            "confidence_low_nepali": "यो पचास रुपैयाँको नोट हुनसक्छ, तर विश्वास कम छ।",
        },
    },
    "rs100": {
        "value": 100,
        "label": "rs100",
        "english_name": "One Hundred Rupees",
        "nepali_name": "एक सय रुपैयाँ",
        "nepali_numeral": "१००",
        "color_hint": "green",
        "voice": {
            "english": "This is a one hundred rupee note. Denomination: one hundred Nepali rupees.",
            "nepali": "यो एक सय रुपैयाँको नोट हो। मूल्य: एक सय नेपाली रुपैयाँ।",
            "mixed": "Detected denomination: 100 rupees. नेपालीमा: एक सय रुपैयाँको नोट।",
            "confidence_high": "Recognized value is 100 rupees with high confidence.",
            "confidence_low": "This might be a 100 rupee note, but confidence is low. Please try a clearer image.",
            "confidence_high_nepali": "एक सय रुपैयाँको नोट भनेर उच्च विश्वासका साथ पहिचान भयो।",
            "confidence_low_nepali": "यो एक सय रुपैयाँको नोट हुनसक्छ, तर विश्वास कम छ।",
        },
    },
    "rs500": {
        "value": 500,
        "label": "rs500",
        "english_name": "Five Hundred Rupees",
        "nepali_name": "पाँच सय रुपैयाँ",
        "nepali_numeral": "५००",
        "color_hint": "red/brown",
        "voice": {
            "english": "This is a five hundred rupee note. Denomination: five hundred Nepali rupees.",
            "nepali": "यो पाँच सय रुपैयाँको नोट हो। मूल्य: पाँच सय नेपाली रुपैयाँ।",
            "mixed": "Detected denomination: 500 rupees. नेपालीमा: पाँच सय रुपैयाँको नोट।",
            "confidence_high": "Recognized value is 500 rupees with high confidence.",
            "confidence_low": "This might be a 500 rupee note, but confidence is low. Please try a clearer image.",
            "confidence_high_nepali": "पाँच सय रुपैयाँको नोट भनेर उच्च विश्वासका साथ पहिचान भयो।",
            "confidence_low_nepali": "यो पाँच सय रुपैयाँको नोट हुनसक्छ, तर विश्वास कम छ।",
        },
    },
    "rs1000": {
        "value": 1000,
        "label": "rs1000",
        "english_name": "One Thousand Rupees",
        "nepali_name": "एक हजार रुपैयाँ",
        "nepali_numeral": "१०००",
        "color_hint": "grey/blue",
        "voice": {
            "english": "This is a one thousand rupee note. Denomination: one thousand Nepali rupees.",
            "nepali": "यो एक हजार रुपैयाँको नोट हो। मूल्य: एक हजार नेपाली रुपैयाँ।",
            "mixed": "Detected denomination: 1000 rupees. नेपालीमा: एक हजार रुपैयाँको नोट।",
            "confidence_high": "Recognized value is 1000 rupees with high confidence.",
            "confidence_low": "This might be a 1000 rupee note, but confidence is low. Please try a clearer image.",
            "confidence_high_nepali": "एक हजार रुपैयाँको नोट भनेर उच्च विश्वासका साथ पहिचान भयो।",
            "confidence_low_nepali": "यो एक हजार रुपैयाँको नोट हुनसक्छ, तर विश्वास कम छ।",
        },
    },
}


def get_note_metadata(label: str) -> dict:
    """Get full metadata for a denomination label. Returns 'unknown' entry for unrecognized labels."""
    if label in NOTE_METADATA:
        return NOTE_METADATA[label].copy()
    return NOTE_METADATA["unknown"].copy()


def get_note_voice_text(label: str, mode: str = "english", confidence: float = 100.0) -> str:
    """
    Get the appropriate voice text for a denomination.

    Args:
        label: denomination label (e.g. 'rs100') or 'unknown'
        mode: 'english', 'nepali', 'mixed', or 'confidence'
        confidence: prediction confidence percentage
    """
    meta = get_note_metadata(label)
    voice = meta["voice"]

    if mode == "nepali":
        return voice["nepali"]
    elif mode == "mixed":
        return voice["mixed"]
    elif mode == "confidence":
        if confidence >= 75:
            return voice["confidence_high"]
        else:
            return voice["confidence_low"]
    else:  # english (default)
        return voice["english"]
