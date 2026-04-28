import re
import inflect
from num2words import num2words

_inflect = inflect.engine()

# --- Abbreviation map ---
# Add more as you encounter them during testing
ABBREVIATIONS = {
    "mr.": "mister",
    "mrs.": "missus",
    "dr.": "doctor",
    "prof.": "professor",
    "sr.": "senior",
    "jr.": "junior",
    "vs.": "versus",
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "fig.": "figure",
    "approx.": "approximately",
    "dept.": "department",
    "govt.": "government",
    "ave.": "avenue",
    "blvd.": "boulevard",
    "st.": "street",
    "nasa": "NASA",
    "fbi": "FBI",
    "cia": "CIA",
    "usa": "USA",
    "uk": "UK",
    "ai": "AI",
    "ml": "ML",
}


def _expand_currency(match: re.Match) -> str:
    """Convert $1.5M → one point five million dollars"""
    symbol = match.group(1)
    amount_str = match.group(2).replace(",", "")
    suffix = match.group(3).upper() if match.group(3) else ""

    currency_map = {"$": "dollar", "€": "euro", "£": "pound", "₹": "rupee"}
    currency = currency_map.get(symbol, "dollar")

    multiplier_map = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}

    try:
        amount = float(amount_str)
        if suffix in multiplier_map:
            amount *= multiplier_map[suffix]
        amount_int = int(amount)
        remainder = amount - amount_int

        if remainder > 0:
            # e.g. 1.5 → "one point five"
            decimal_part = str(round(remainder, 4)).split(".")[1].rstrip("0")
            spoken = num2words(amount_int) + " point " + " ".join(num2words(int(d)) for d in decimal_part)
        else:
            spoken = num2words(amount_int)

        # pluralize currency
        if amount != 1:
            currency = _inflect.plural(currency)

        return spoken + " " + currency
    except (ValueError, TypeError):
        return match.group(0)


def _expand_number(match: re.Match) -> str:
    """Convert standalone numbers to words, with special handling for years."""
    num_str = match.group(0).replace(",", "")
    try:
        if "." in num_str:
            return num2words(float(num_str))

        value = int(num_str)

        # Year detection: 4-digit numbers between 1000 and 2099
        # Spoken as pairs: 2023 → "twenty twenty-three", 1995 → "nineteen ninety-five"
        if 1000 <= value <= 2099:
            century = value // 100      # 20
            remainder = value % 100     # 23
            if remainder == 0:
                # e.g. 2000 → "two thousand", 1900 → "nineteen hundred"
                return num2words(value)
            century_word = num2words(century)    # "twenty"
            remainder_word = num2words(remainder) # "twenty-three"
            return f"{century_word} {remainder_word}"

        return num2words(value)

    except (ValueError, TypeError):
        return num_str

def _expand_abbreviations(text: str) -> str:
    """Expand known abbreviations. Case-insensitive lookup."""
    words = text.split()
    expanded = []
    for word in words:
        lower = word.lower()
        if lower in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[lower])
        else:
            expanded.append(word)
    return " ".join(expanded)


def _remove_special_chars(text: str) -> str:
    """Strip characters TTS models choke on, keep punctuation that affects prosody."""
    # Keep: letters, digits, spaces, and prosodic punctuation
    text = re.sub(r"[^\w\s.,!?;:\'\"-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)
    return text.strip()


def normalize(text: str) -> str:
    """
    Full normalization pipeline.
    Input:  raw user text e.g. "Dr. Smith earned $1.5M in 2023."
    Output: TTS-ready text  e.g. "doctor Smith earned one point five million dollars in two thousand and twenty three."
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    # 1. Strip leading/trailing whitespace
    text = text.strip()

    # 2. Expand currency  e.g.  $1.5M  £200  €3,000
    text = re.sub(
        r"([\$€£₹])([\d,]+\.?\d*)([KMBTkmbt]?)",
        _expand_currency,
        text
    )

    # 3. Expand abbreviations
    text = _expand_abbreviations(text)

    # 4. Expand standalone numbers (after currency so we don't double-process)
    text = re.sub(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", _expand_number, text)

    # 5. Remove characters the model can't handle
    text = _remove_special_chars(text)

    return text