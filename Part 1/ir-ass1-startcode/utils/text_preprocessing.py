'''
Robust HTML → tokens cleaning pipeline (Task A-1).

This module turns messy web HTML into clean token lists suitable for indexing.
Key choices:
- Strip <script>/<style>/<noscript> to remove non-content noise.
- Decode HTML entities early (&nbsp;, &amp;) so downstream tokenisation sees text.
- Unicode normalisation (NFKC) to fold smart quotes/full-width forms.
- Remove zero-width/control chars; collapse whitespace to single spaces.
- Keep punctuation (tokenised separately) no stemming/stopwords per Part A spec.

Design goal is to be tolerant of malformed HTML and odd encodings without crashing,
and produce stable input for IR components.
'''

import re, html, unicodedata
from typing import List
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  

lemmatizer = WordNetLemmatizer()  # unused 

__all__ = ["preprocess"]


def _ensure_punkt() -> None:
    """make sure NLTK tokenizers are available 
        handles older NLTK gracefully)."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            # older NLTK versions don't have punkt_tab safe to ignore.
            pass


def _clean_html(raw_html: str) -> str:
    """strip tags/scripts/styles, decode entities, normalise unicode & whitespace."""
    html_input = "" if raw_html is None else str(raw_html)
    html_input = html.unescape(html_input)
    soup = BeautifulSoup(html_input, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\u2423", " ")
    text = text.replace("\u00A0", " ")
    text = text.replace("\u00AD", "")

    text = re.sub(r"[\u200B-\u200D\uFEFF]", " ", text)         # zero-width -> space
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)  # control chars

    text = re.sub(r'(?<=\w)&(?=\w)', '§AMP§', text)
    text = re.sub(r'\s*&\s*', ' and ', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(raw_html_list: List[str]) -> List[List[str]]:
    """Convert noisy HTML documents into token lists using NLTK.

    Input: list of raw HTML strings (None allowed; treated as empty).
    Output: list of token lists (punctuation kept as separate tokens).

    Failure-tolerance: malformed HTML, odd encodings, and missing data should
    not raise; instead degrade gracefully to empty outputs.
    """
    
    if not isinstance(raw_html_list, list):
        raise TypeError("preprocess expects a list of HTML strings")

    _ensure_punkt()

    outputs: List[List[str]] = []
    for raw in raw_html_list:
        text = _clean_html(raw)
        tokens = word_tokenize(text) if text else []
        # Restore protected ampersands
        if tokens:
            tokens = [t.replace('§AMP§', '&') for t in tokens]
        outputs.append(tokens)
    return outputs



