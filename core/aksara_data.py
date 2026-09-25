# File: core/aksara_data.py
"""
Data Metadata dan Mapping Unicode untuk 120 Kelas Aksara Jawa
Meliputi 20 Aksara Dasar (Nglegena) + 5 Sandhangan Swara (Wulu, Suku, Taling, Pepet, Taling-Tarung)
"""

BASE_CONSONANTS = {
    'ha': {'char': '\uA9B2', 'latin': 'Ha', 'name': 'Aksara Ha', 'meaning': 'Hana (Ada)'},
    'na': {'char': '\uA9A4', 'latin': 'Na', 'name': 'Aksara Na', 'meaning': 'Caraka (Utusan)'},
    'ca': {'char': '\uA995', 'latin': 'Ca', 'name': 'Aksara Ca', 'meaning': 'Caraka (Utusan)'},
    'ra': {'char': '\uA9AB', 'latin': 'Ra', 'name': 'Aksara Ra', 'meaning': 'Caraka (Utusan)'},
    'ka': {'char': '\uA98F', 'latin': 'Ka', 'name': 'Aksara Ka', 'meaning': 'Caraka (Utusan)'},
    'da': {'char': '\uA9A2', 'latin': 'Da', 'name': 'Aksara Da', 'meaning': 'Datasawala (Berbeda/Setara)'},
    'ta': {'char': '\uA9A0', 'latin': 'Ta', 'name': 'Aksara Ta', 'meaning': 'Datasawala (Berbeda/Setara)'},
    'sa': {'char': '\uA9B1', 'latin': 'Sa', 'name': 'Aksara Sa', 'meaning': 'Datasawala (Berbeda/Setara)'},
    'wa': {'char': '\uA9AE', 'latin': 'Wa', 'name': 'Aksara Wa', 'meaning': 'Datasawala (Berbeda/Setara)'},
    'la': {'char': '\uA9AD', 'latin': 'La', 'name': 'Aksara La', 'meaning': 'Datasawala (Berbeda/Setara)'},
    'pa': {'char': '\uA9A5', 'latin': 'Pa', 'name': 'Aksara Pa', 'meaning': 'Padhajayanya (Sama Kuatnya)'},
    'dha': {'char': '\uA99D', 'latin': 'Dha', 'name': 'Aksara Dha', 'meaning': 'Padhajayanya (Sama Kuatnya)'},
    'ja': {'char': '\uA997', 'latin': 'Ja', 'name': 'Aksara Ja', 'meaning': 'Padhajayanya (Sama Kuatnya)'},
    'ya': {'char': '\uA9AA', 'latin': 'Ya', 'name': 'Aksara Ya', 'meaning': 'Padhajayanya (Sama Kuatnya)'},
    'nya': {'char': '\uA99A', 'latin': 'Nya', 'name': 'Aksara Nya', 'meaning': 'Padhajayanya (Sama Kuatnya)'},
    'ma': {'char': '\uA9A9', 'latin': 'Ma', 'name': 'Aksara Ma', 'meaning': 'Magabathanga (Menjadi Mayat)'},
    'ga': {'char': '\uA992', 'latin': 'Ga', 'name': 'Aksara Ga', 'meaning': 'Magabathanga (Menjadi Mayat)'},
    'ba': {'char': '\uA9A7', 'latin': 'Ba', 'name': 'Aksara Ba', 'meaning': 'Magabathanga (Menjadi Mayat)'},
    'tha': {'char': '\uA99B', 'latin': 'Tha', 'name': 'Aksara Tha', 'meaning': 'Magabathanga (Menjadi Mayat)'},
    'nga': {'char': '\uA994', 'latin': 'Nga', 'name': 'Aksara Nga', 'meaning': 'Magabathanga (Menjadi Mayat)'},
}

CONSONANT_PREFIX_MAP = {
    'dha': 'dha', 'dh': 'dha',
    'tha': 'tha', 'th': 'tha',
    'nga': 'nga', 'ng': 'nga',
    'nya': 'nya', 'ny': 'nya',
    'ha': 'ha', 'h': 'ha',
    'na': 'na', 'n': 'na',
    'ca': 'ca', 'c': 'ca',
    'ra': 'ra', 'r': 'ra',
    'ka': 'ka', 'k': 'ka',
    'da': 'da', 'd': 'da',
    'ta': 'ta', 't': 'ta',
    'sa': 'sa', 's': 'sa',
    'wa': 'wa', 'w': 'wa',
    'la': 'la', 'l': 'la',
    'pa': 'pa', 'p': 'pa',
    'ja': 'ja', 'j': 'ja',
    'ya': 'ya', 'y': 'ya',
    'ma': 'ma', 'm': 'ma',
    'ga': 'ga', 'g': 'ga',
    'ba': 'ba', 'b': 'ba',
}

SANDHANGAN_INFO = {
    'aksara-dasar': {
        'name': 'Aksara Dasar (Nglegena)',
        'vowel': 'a',
        'symbol': '',
        'position': 'Bentuk Baku',
        'desc': 'Aksara dasar nglegena bersuara terbuka /a/.'
    },
    'wulu': {
        'name': 'Sandhangan Wulu',
        'vowel': 'i',
        'symbol': '\uA9B6',
        'position': 'Di atas aksara',
        'desc': 'Mengubah bunyi vokal aksara menjadi /i/.'
    },
    'suku': {
        'name': 'Sandhangan Suku',
        'vowel': 'u',
        'symbol': '\uA9B8',
        'position': 'Di bawah kanan aksara',
        'desc': 'Mengubah bunyi vokal aksara menjadi /u/.'
    },
    'taling': {
        'name': 'Sandhangan Taling',
        'vowel': 'é',
        'symbol': '\uA9BA',
        'position': 'Di depan (kiri) aksara',
        'desc': 'Mengubah bunyi vokal aksara menjadi /é/ (taling miring/jejeg).'
    },
    'pepet': {
        'name': 'Sandhangan Pepet',
        'vowel': 'ê',
        'symbol': '\uA9BC',
        'position': 'Di atas aksara',
        'desc': 'Mengubah bunyi vokal aksara menjadi /ê/ (vokal sedang/pepet).'
    },
    'taling-tarung': {
        'name': 'Sandhangan Taling-Tarung',
        'vowel': 'o',
        'symbol': '\uA9BA\uA9B4',
        'position': 'Mengapit (depan & belakang)',
        'desc': 'Mengubah bunyi vokal aksara menjadi /o/.'
    }
}


def clean_consonant_key(letter: str, category: str) -> str:
    """Mengekstrak kunci konsonan dasar (ha, na, ca, dll) dari nama label."""
    s = letter.lower()
    for ending in ['ê', 'ê', 'o', 'u', 'i', 'e', 'a']:
        if s.endswith(ending):
            s = s[:-len(ending)]
            break
    return CONSONANT_PREFIX_MAP.get(s, s)


def get_aksara_details(class_name: str) -> dict:
    """Mengembalikan informasi lengkap metadata dan Unicode untuk satu nama kelas."""
    parts = class_name.split('_')
    if len(parts) != 2:
        return {
            'class_name': class_name,
            'category': 'Unknown',
            'latin': class_name,
            'unicode_char': 'ꦄ',
            'category_name': 'Unknown',
            'desc': 'Aksara Jawa'
        }

    category, letter = parts[0], parts[1]
    sandhangan = SANDHANGAN_INFO.get(category, SANDHANGAN_INFO['aksara-dasar'])
    consonant_key = clean_consonant_key(letter, category)
    base = BASE_CONSONANTS.get(consonant_key, BASE_CONSONANTS['ha'])

    base_char = base['char']
    base_name = base['latin']

    # Bangun karakter Unicode
    if category == 'aksara-dasar':
        unicode_char = base_char
        latin_translit = base_name
        description = f"Aksara dasar {base_name} (vokal 'a')."
    elif category == 'wulu':
        unicode_char = base_char + '\uA9B6'
        latin_translit = f"{base_name[:-1]}i" if base_name.endswith('a') else f"{base_name}i"
        description = f"Aksara {base_name} diberi Sandhangan Wulu (bunyi 'i')."
    elif category == 'suku':
        unicode_char = base_char + '\uA9B8'
        latin_translit = f"{base_name[:-1]}u" if base_name.endswith('a') else f"{base_name}u"
        description = f"Aksara {base_name} diberi Sandhangan Suku (bunyi 'u')."
    elif category == 'taling':
        unicode_char = base_char + '\uA9BA'
        latin_translit = f"{base_name[:-1]}é" if base_name.endswith('a') else f"{base_name}é"
        description = f"Aksara {base_name} diberi Sandhangan Taling (bunyi 'é')."
    elif category == 'pepet':
        unicode_char = base_char + '\uA9BC'
        latin_translit = f"{base_name[:-1]}ê" if base_name.endswith('a') else f"{base_name}ê"
        description = f"Aksara {base_name} diberi Sandhangan Pepet (bunyi 'ê')."
    elif category == 'taling-tarung':
        unicode_char = base_char + '\uA9BA\uA9B4'
        latin_translit = f"{base_name[:-1]}o" if base_name.endswith('a') else f"{base_name}o"
        description = f"Aksara {base_name} diberi Sandhangan Taling-Tarung (bunyi 'o')."
    else:
        unicode_char = base_char
        latin_translit = letter.capitalize()
        description = f"Aksara {class_name}"

    return {
        'class_name': class_name,
        'category': category,
        'category_name': sandhangan['name'],
        'vowel': sandhangan['vowel'],
        'position': sandhangan['position'],
        'base_consonant': base_name,
        'latin': latin_translit,
        'unicode_char': unicode_char,
        'desc': description
    }
