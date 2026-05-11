import pickle, pathlib, gzip
from typing import Any

#modify this file to suit needs, this serilization is only for demonstration purposes

def dump(obj: Any, path: str):
    path = pathlib.Path(path)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
