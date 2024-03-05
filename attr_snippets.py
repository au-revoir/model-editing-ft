import collections
import json 
import torch
from pathlib import Path
            
class AttributeSnippets:
    """ 
    Contains wikipedia snippets discussing entities that have some property.
    More formally, given a tuple t = (s, r, o):
    - Let snips = AttributeSnippets(DATA_DIR)
    - snips[r][o] is a list of wikipedia articles for all s' such that t' = (s', r, o) is valid.
    """
    def __init__(self, snips_loc):
        with open(snips_loc, "r") as f:
            snippets_list = json.load(f)

        snips = collections.defaultdict(lambda: collections.defaultdict(list))

        for el in snippets_list:
            rid, tid = el["relation_id"], el["target_id"]
            for sample in el["samples"]:
                snips[rid][tid].append(sample)

        self._data = snips
        self.snippets_list = snippets_list

    def __getitem__(self, item):
        return self._data[item]
