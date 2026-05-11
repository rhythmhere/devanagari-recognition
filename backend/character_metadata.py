"""
Comprehensive Devanagari character metadata with educational information.
Maps raw model labels to rich character data including transliteration,
Nepali names, types, example words, notes, and common confusions.
"""

CHAR_METADATA = {
    # ── Vowels (स्वर) ──────────────────────────────────────────────
    "1_a": {
        "char": "अ",
        "roman": "a",
        "nepali_name": "अ",
        "type": "vowel",
        "example_word": "अनार (anaar) – pomegranate",
        "note": "First letter of the Devanagari script. Represents the short 'a' sound.",
        "confusions": ["2_aa", "12_am"]
    },
    "2_aa": {
        "char": "आ",
        "roman": "aa",
        "nepali_name": "आ",
        "type": "vowel",
        "example_word": "आमा (aamaa) – mother",
        "note": "Long vowel 'aa'. Extended version of अ.",
        "confusions": ["1_a", "12_am"]
    },
    "3_e": {
        "char": "इ",
        "roman": "i",
        "nepali_name": "इ",
        "type": "vowel",
        "example_word": "इनार (inaar) – well",
        "note": "Short vowel 'i'.",
        "confusions": ["4_ee"]
    },
    "4_ee": {
        "char": "ई",
        "roman": "ee",
        "nepali_name": "ई",
        "type": "vowel",
        "example_word": "ईख (eekh) – sugarcane",
        "note": "Long vowel 'ee'. Extended version of इ.",
        "confusions": ["3_e"]
    },
    "5_u": {
        "char": "उ",
        "roman": "u",
        "nepali_name": "उ",
        "type": "vowel",
        "example_word": "उन (un) – wool",
        "note": "Short vowel 'u'.",
        "confusions": ["6_uu"]
    },
    "6_uu": {
        "char": "ऊ",
        "roman": "uu",
        "nepali_name": "ऊ",
        "type": "vowel",
        "example_word": "ऊन (oon) – wool",
        "note": "Long vowel 'oo'. Extended version of उ.",
        "confusions": ["5_u"]
    },
    "7_ru": {
        "char": "ऋ",
        "roman": "ri",
        "nepali_name": "ऋ",
        "type": "vowel",
        "example_word": "ऋषि (rishi) – sage",
        "note": "Vowel representing the 'ri' sound. Rare in modern Nepali.",
        "confusions": []
    },
    "8_ye": {
        "char": "ए",
        "roman": "e",
        "nepali_name": "ए",
        "type": "vowel",
        "example_word": "एकता (ekataa) – unity",
        "note": "Vowel 'e' as in 'play'.",
        "confusions": ["9_i"]
    },
    "9_i": {
        "char": "ऐ",
        "roman": "ai",
        "nepali_name": "ऐ",
        "type": "vowel",
        "example_word": "ऐना (ainaa) – mirror",
        "note": "Diphthong vowel 'ai'.",
        "confusions": ["8_ye"]
    },
    "10_o": {
        "char": "ओ",
        "roman": "o",
        "nepali_name": "ओ",
        "type": "vowel",
        "example_word": "ओखर (okhar) – walnut",
        "note": "Vowel 'o' as in 'go'.",
        "confusions": ["11_au"]
    },
    "11_au": {
        "char": "औ",
        "roman": "au",
        "nepali_name": "औ",
        "type": "vowel",
        "example_word": "औषधि (aushadhi) – medicine",
        "note": "Diphthong vowel 'au'.",
        "confusions": ["10_o"]
    },
    "12_am": {
        "char": "अं",
        "roman": "am",
        "nepali_name": "अं (अनुस्वार)",
        "type": "vowel",
        "example_word": "अंश (ansha) – portion",
        "note": "Anusvara – nasal sound marker placed above a character.",
        "confusions": ["1_a", "13_aha"]
    },
    "13_aha": {
        "char": "अः",
        "roman": "aha",
        "nepali_name": "अः (विसर्ग)",
        "type": "vowel",
        "example_word": "दुःख (dukha) – sorrow",
        "note": "Visarga – aspirated breath sound, appears as two dots.",
        "confusions": ["12_am"]
    },

    # ── Consonants (व्यञ्जन) ────────────────────────────────────────
    "character_1_ka": {
        "char": "क",
        "roman": "ka",
        "nepali_name": "क",
        "type": "consonant",
        "example_word": "कलम (kalam) – pen",
        "note": "First consonant (ka-varga). Velar stop.",
        "confusions": ["character_2_kha", "character_22_pha"]
    },
    "character_2_kha": {
        "char": "ख",
        "roman": "kha",
        "nepali_name": "ख",
        "type": "consonant",
        "example_word": "खबर (khabar) – news",
        "note": "Aspirated 'ka'. Same group as क.",
        "confusions": ["character_1_ka"]
    },
    "character_3_ga": {
        "char": "ग",
        "roman": "ga",
        "nepali_name": "ग",
        "type": "consonant",
        "example_word": "गाई (gaai) – cow",
        "note": "Voiced velar stop.",
        "confusions": ["character_4_gha"]
    },
    "character_4_gha": {
        "char": "घ",
        "roman": "gha",
        "nepali_name": "घ",
        "type": "consonant",
        "example_word": "घर (ghar) – house",
        "note": "Aspirated voiced velar stop.",
        "confusions": ["character_3_ga", "character_19_dha"]
    },
    "character_5_kna": {
        "char": "ङ",
        "roman": "nga",
        "nepali_name": "ङ",
        "type": "consonant",
        "example_word": "गंगा (gangaa) – Ganges river",
        "note": "Velar nasal. Rarely used independently in modern Nepali.",
        "confusions": []
    },
    "character_6_cha": {
        "char": "च",
        "roman": "cha",
        "nepali_name": "च",
        "type": "consonant",
        "example_word": "चम्चा (chamchaa) – spoon",
        "note": "Palatal affricate.",
        "confusions": ["character_7_chha"]
    },
    "character_7_chha": {
        "char": "छ",
        "roman": "chha",
        "nepali_name": "छ",
        "type": "consonant",
        "example_word": "छाता (chhaataa) – umbrella",
        "note": "Aspirated palatal affricate.",
        "confusions": ["character_6_cha"]
    },
    "character_8_ja": {
        "char": "ज",
        "roman": "ja",
        "nepali_name": "ज",
        "type": "consonant",
        "example_word": "जल (jal) – water",
        "note": "Voiced palatal affricate.",
        "confusions": ["character_9_jha"]
    },
    "character_9_jha": {
        "char": "झ",
        "roman": "jha",
        "nepali_name": "झ",
        "type": "consonant",
        "example_word": "झण्डा (jhandaa) – flag",
        "note": "Aspirated voiced palatal affricate.",
        "confusions": ["character_8_ja"]
    },
    "character_10_yna": {
        "char": "ञ",
        "roman": "nya",
        "nepali_name": "ञ",
        "type": "consonant",
        "example_word": "ज्ञान (gyaan) – knowledge",
        "note": "Palatal nasal. Rarely used alone.",
        "confusions": []
    },
    "character_11_taamatar": {
        "char": "ट",
        "roman": "tta",
        "nepali_name": "ट",
        "type": "consonant",
        "example_word": "टोपी (topee) – cap",
        "note": "Retroflex unaspirated stop.",
        "confusions": ["character_12_thaa", "character_16_tabala"]
    },
    "character_12_thaa": {
        "char": "ठ",
        "roman": "ttha",
        "nepali_name": "ठ",
        "type": "consonant",
        "example_word": "ठूलो (thulo) – big",
        "note": "Retroflex aspirated stop.",
        "confusions": ["character_11_taamatar"]
    },
    "character_13_daa": {
        "char": "ड",
        "roman": "dda",
        "nepali_name": "ड",
        "type": "consonant",
        "example_word": "डमरु (damaru) – drum",
        "note": "Retroflex voiced stop.",
        "confusions": ["character_14_dhaa"]
    },
    "character_14_dhaa": {
        "char": "ढ",
        "roman": "ddha",
        "nepali_name": "ढ",
        "type": "consonant",
        "example_word": "ढोका (dhokaa) – door",
        "note": "Retroflex voiced aspirated stop.",
        "confusions": ["character_13_daa", "character_4_gha"]
    },
    "character_15_adna": {
        "char": "ण",
        "roman": "nna",
        "nepali_name": "ण",
        "type": "consonant",
        "example_word": "गुण (guna) – quality",
        "note": "Retroflex nasal.",
        "confusions": ["character_20_na"]
    },
    "character_16_tabala": {
        "char": "त",
        "roman": "ta",
        "nepali_name": "त",
        "type": "consonant",
        "example_word": "तरकारी (tarkaari) – vegetable",
        "note": "Dental unaspirated stop.",
        "confusions": ["character_17_tha", "character_11_taamatar"]
    },
    "character_17_tha": {
        "char": "थ",
        "roman": "tha",
        "nepali_name": "थ",
        "type": "consonant",
        "example_word": "थाली (thaali) – plate",
        "note": "Dental aspirated stop.",
        "confusions": ["character_16_tabala", "character_12_thaa"]
    },
    "character_18_da": {
        "char": "द",
        "roman": "da",
        "nepali_name": "द",
        "type": "consonant",
        "example_word": "दाल (daal) – lentil",
        "note": "Dental voiced stop.",
        "confusions": ["character_19_dha"]
    },
    "character_19_dha": {
        "char": "ध",
        "roman": "dha",
        "nepali_name": "ध",
        "type": "consonant",
        "example_word": "धन (dhan) – wealth",
        "note": "Dental voiced aspirated stop.",
        "confusions": ["character_18_da", "character_4_gha"]
    },
    "character_20_na": {
        "char": "न",
        "roman": "na",
        "nepali_name": "न",
        "type": "consonant",
        "example_word": "नमस्ते (namaste) – greeting",
        "note": "Dental nasal. One of the most common consonants.",
        "confusions": ["character_15_adna"]
    },
    "character_21_pa": {
        "char": "प",
        "roman": "pa",
        "nepali_name": "प",
        "type": "consonant",
        "example_word": "पानी (paani) – water",
        "note": "Bilabial unaspirated stop.",
        "confusions": ["character_22_pha"]
    },
    "character_22_pha": {
        "char": "फ",
        "roman": "pha",
        "nepali_name": "फ",
        "type": "consonant",
        "example_word": "फूल (phool) – flower",
        "note": "Bilabial aspirated stop.",
        "confusions": ["character_21_pa"]
    },
    "character_23_ba": {
        "char": "ब",
        "roman": "ba",
        "nepali_name": "ब",
        "type": "consonant",
        "example_word": "बाटो (baato) – road",
        "note": "Bilabial voiced stop.",
        "confusions": ["character_24_bha", "character_29_waw"]
    },
    "character_24_bha": {
        "char": "भ",
        "roman": "bha",
        "nepali_name": "भ",
        "type": "consonant",
        "example_word": "भात (bhaat) – rice",
        "note": "Bilabial voiced aspirated stop.",
        "confusions": ["character_23_ba"]
    },
    "character_25_ma": {
        "char": "म",
        "roman": "ma",
        "nepali_name": "म",
        "type": "consonant",
        "example_word": "माछा (maachaa) – fish",
        "note": "Bilabial nasal.",
        "confusions": ["character_24_bha"]
    },
    "character_26_yaw": {
        "char": "य",
        "roman": "ya",
        "nepali_name": "य",
        "type": "consonant",
        "example_word": "यात्रा (yaatraa) – journey",
        "note": "Palatal approximant.",
        "confusions": []
    },
    "character_27_ra": {
        "char": "र",
        "roman": "ra",
        "nepali_name": "र",
        "type": "consonant",
        "example_word": "राम (raam) – name / lord",
        "note": "Alveolar tap/trill. Very common in Nepali.",
        "confusions": ["character_28_la"]
    },
    "character_28_la": {
        "char": "ल",
        "roman": "la",
        "nepali_name": "ल",
        "type": "consonant",
        "example_word": "लामो (laamo) – long",
        "note": "Alveolar lateral.",
        "confusions": ["character_27_ra"]
    },
    "character_29_waw": {
        "char": "व",
        "roman": "wa",
        "nepali_name": "व",
        "type": "consonant",
        "example_word": "वन (ban) – forest",
        "note": "Labio-dental approximant. Pronounced as 'ba' or 'wa'.",
        "confusions": ["character_23_ba"]
    },
    "character_30_motosaw": {
        "char": "श",
        "roman": "sha",
        "nepali_name": "श",
        "type": "consonant",
        "example_word": "शहर (shahar) – city",
        "note": "Palatal fricative 'sh'.",
        "confusions": ["character_31_petchiryakha", "character_32_patalosaw"]
    },
    "character_31_petchiryakha": {
        "char": "ष",
        "roman": "ssa",
        "nepali_name": "ष",
        "type": "consonant",
        "example_word": "षट्कोण (shatkona) – hexagon",
        "note": "Retroflex sibilant. Rare in spoken Nepali.",
        "confusions": ["character_30_motosaw", "character_32_patalosaw"]
    },
    "character_32_patalosaw": {
        "char": "स",
        "roman": "sa",
        "nepali_name": "स",
        "type": "consonant",
        "example_word": "समय (samaya) – time",
        "note": "Dental sibilant 's'. Very common.",
        "confusions": ["character_30_motosaw"]
    },
    "character_33_ha": {
        "char": "ह",
        "roman": "ha",
        "nepali_name": "ह",
        "type": "consonant",
        "example_word": "हात (haat) – hand",
        "note": "Glottal fricative. Last pure consonant in the alphabet.",
        "confusions": []
    },
    "character_34_chhya": {
        "char": "क्ष",
        "roman": "kshya",
        "nepali_name": "क्ष",
        "type": "consonant",
        "example_word": "क्षमा (kshamaa) – forgiveness",
        "note": "Compound consonant (क + ष). Used in Sanskrit-origin words.",
        "confusions": ["character_35_tra", "character_36_gya"]
    },
    "character_35_tra": {
        "char": "त्र",
        "roman": "tra",
        "nepali_name": "त्र",
        "type": "consonant",
        "example_word": "त्रिभुवन (tribhuvan) – three worlds",
        "note": "Compound consonant (त + र).",
        "confusions": ["character_34_chhya", "character_36_gya"]
    },
    "character_36_gya": {
        "char": "ज्ञ",
        "roman": "gya",
        "nepali_name": "ज्ञ",
        "type": "consonant",
        "example_word": "ज्ञान (gyaan) – knowledge",
        "note": "Compound consonant (ज + ञ). Pronounced 'gya' in Nepali.",
        "confusions": ["character_34_chhya", "character_35_tra"]
    },

    # ── Digits (अंक) ───────────────────────────────────────────────
    "digit_0": {
        "char": "०",
        "roman": "0",
        "nepali_name": "शून्य",
        "type": "digit",
        "example_word": "शून्य (shunya) – zero",
        "note": "Devanagari digit zero.",
        "confusions": ["digit_9"]
    },
    "digit_1": {
        "char": "१",
        "roman": "1",
        "nepali_name": "एक",
        "type": "digit",
        "example_word": "एक (ek) – one",
        "note": "Devanagari digit one.",
        "confusions": ["digit_2"]
    },
    "digit_2": {
        "char": "२",
        "roman": "2",
        "nepali_name": "दुई",
        "type": "digit",
        "example_word": "दुई (dui) – two",
        "note": "Devanagari digit two.",
        "confusions": ["digit_1", "digit_3"]
    },
    "digit_3": {
        "char": "३",
        "roman": "3",
        "nepali_name": "तीन",
        "type": "digit",
        "example_word": "तीन (teen) – three",
        "note": "Devanagari digit three.",
        "confusions": ["digit_2"]
    },
    "digit_4": {
        "char": "४",
        "roman": "4",
        "nepali_name": "चार",
        "type": "digit",
        "example_word": "चार (chaar) – four",
        "note": "Devanagari digit four.",
        "confusions": ["digit_8"]
    },
    "digit_5": {
        "char": "५",
        "roman": "5",
        "nepali_name": "पाँच",
        "type": "digit",
        "example_word": "पाँच (paanch) – five",
        "note": "Devanagari digit five.",
        "confusions": []
    },
    "digit_6": {
        "char": "६",
        "roman": "6",
        "nepali_name": "छ",
        "type": "digit",
        "example_word": "छ (chha) – six",
        "note": "Devanagari digit six.",
        "confusions": ["digit_9"]
    },
    "digit_7": {
        "char": "७",
        "roman": "7",
        "nepali_name": "सात",
        "type": "digit",
        "example_word": "सात (saat) – seven",
        "note": "Devanagari digit seven.",
        "confusions": []
    },
    "digit_8": {
        "char": "८",
        "roman": "8",
        "nepali_name": "आठ",
        "type": "digit",
        "example_word": "आठ (aath) – eight",
        "note": "Devanagari digit eight.",
        "confusions": ["digit_4"]
    },
    "digit_9": {
        "char": "९",
        "roman": "9",
        "nepali_name": "नौ",
        "type": "digit",
        "example_word": "नौ (nau) – nine",
        "note": "Devanagari digit nine.",
        "confusions": ["digit_0", "digit_6"]
    },
}


def get_metadata(raw_label: str) -> dict:
    """Get rich metadata for a raw model label."""
    meta = CHAR_METADATA.get(raw_label, None)
    if meta:
        return meta
    # Fallback for unknown labels
    return {
        "char": raw_label,
        "roman": raw_label,
        "nepali_name": raw_label,
        "type": "unknown",
        "example_word": "",
        "note": "",
        "confusions": []
    }


def get_confusion_chars(raw_label: str) -> list:
    """Get list of characters this label may be confused with."""
    meta = CHAR_METADATA.get(raw_label, {})
    confusion_labels = meta.get("confusions", [])
    result = []
    for cl in confusion_labels:
        cm = CHAR_METADATA.get(cl, {})
        if cm:
            result.append({
                "raw_label": cl,
                "char": cm["char"],
                "roman": cm["roman"]
            })
    return result
