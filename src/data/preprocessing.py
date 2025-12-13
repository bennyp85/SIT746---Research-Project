import codecs
from collections import Counter

def tsf_series_lengths(path):
    lengths = []
    names = []
    with codecs.open(path, encoding="latin-1") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            # Format: series_name : start_timestamp : v1,v2,...,vN
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            name, start_ts, values_str = parts
            values = [v for v in values_str.split(",") if v != ""]
            names.append(name)
            lengths.append(len(values))
    return names, lengths