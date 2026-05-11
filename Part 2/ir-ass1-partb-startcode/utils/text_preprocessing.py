"""Robust HTML→tokens cleaning pipeline (Task A-1)."""
import re, html, unicodedata
from typing import List
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess(raw_html_list: List[str]) -> List[List[str]]:
    """Convert noisy HTML documents into token lists.

    - strips tags & entities
    - keeps 1 punctuation
    """
    pass
