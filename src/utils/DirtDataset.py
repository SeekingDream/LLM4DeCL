import os.path

import torch
import json
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union
from src.utils.dirt.variable import Location, Variable, location_from_json_key, Register, Stack
from src.utils.dirt.dire_types import Struct, TypeLibCodec, TypeLib, UDT, TypeInfo, Disappear
from src.utils.dirt.code_processing import tokenize_raw_code
from src.utils.dirt.function import CollectedFunction, Function


class DirtExample:
    def __init__(
            self,
            name: str,
            code_tokens: str,
            source: Mapping[Location, Set[Variable]],
            target: Mapping[Location, Set[Variable]],
            binary_file: str = "",
            valid: bool = True,
            raw_code: str = "",
            test_meta: Dict[str, Dict[str, bool]] = None,
            binary: str = None,
    ):
        self.name = name
        self.code_tokens = code_tokens
        self.source = source
        self.target = target
        self.binary_file = binary_file
        self._is_valid = valid
        self.raw_code = raw_code
        self.test_meta = test_meta
        self.binary = binary

    @classmethod
    def from_dataline(cls, dataline):
        return cls.from_json_string(dataline['jsonl'])

    @classmethod
    def from_json_string(cls, s: str):
        return cls.from_json(json.loads(s))

    @classmethod
    def from_json(cls, d: Dict):
        source = {
            location_from_json_key(loc): Variable.from_json(var)
            for loc, var in d["source"].items()
        }
        target = {
            location_from_json_key(loc): Variable.from_json(var)
            for loc, var in d["target"].items()
        }
        return cls(
            d["name"],
            d["code_tokens"],
            source,
            target,
            test_meta=d.get("test_meta", None),
            binary=d.get("binary", None),
        )

    def to_json(self):
        assert self._is_valid
        source = {loc.json_key(): var.to_json() for loc, var in self.source.items()}
        target = {loc.json_key(): var.to_json() for loc, var in self.target.items()}
        return {
            "name": self.name,
            "code_tokens": self.code_tokens,
            "source": source,
            "target": target,
        }

    @classmethod
    def from_cf(cls, cf: CollectedFunction, **kwargs):
        """Convert from a decoded CollectedFunction"""
        name = cf.decompiler.name
        raw_code = cf.decompiler.raw_code
        code_tokens = tokenize_raw_code(raw_code)

        source = {**cf.decompiler.local_vars, **cf.decompiler.arguments}
        target = {**cf.debug.local_vars, **cf.debug.arguments}

        # Remove variables that overlap on memory or don't appear in the code tokens
        source_code_tokens_set = set(code_tokens)
        target_code_tokens_set = set(tokenize_raw_code(cf.debug.raw_code))

        source = DirtExample.filter(source, source_code_tokens_set)
        target = DirtExample.filter(target, target_code_tokens_set, set(source.keys()))

        # Assign type "Disappear" to variables not existing in the ground truth
        varnames = set()
        for loc in source.keys():
            if loc not in target.keys():
                target[loc] = Variable(Disappear(), "", False)
        # Add special tokens to variables  to prevnt being sub-tokenized in BPE
        for var in source.values():
            varname = var.name
            varnames.add(varname)
        for idx in range(len(code_tokens)):
            if code_tokens[idx] in varnames:
                code_tokens[idx] = f"@@{code_tokens[idx]}@@"

        return cls(
            name,
            code_tokens,
            source,
            target,
            kwargs["binary_file"],
            valid=name == cf.debug.name and source,
            raw_code=raw_code,
        )

    @staticmethod
    def filter(
            mapping: Mapping[Location, Set[Variable]],
            code_tokens: Optional[Set[str]] = None,
            locations: Optional[Set[Location]] = None,
    ) -> Mapping[Location, Variable]:
        """Discard and leave these for future work:

        Multiple variables sharing a memory location (no way to determine ground truth);
        Variables not appearing in code (no way to get representation);
        Target variables not appearing in source (useless ground truth);
        """
        ret: Mapping[Location, Set[Variable]] = {}
        for location, variable_set in mapping.items():
            if len(variable_set) > 1:
                continue
            var = list(variable_set)[0]
            if code_tokens is not None and not var.name in code_tokens:
                continue
            if locations is not None and not location in locations:
                continue
            ret[location] = var
        return ret

    @property
    def is_valid_example(self):
        return self._is_valid


# HACK: Stupid global lambda functions required for distributed data loading
def identity(x):
    return x


def get_src_len(e):
    return e.source_seq_length


class DirtDataset(torch.utils.data.Dataset):
    SHUFFLE_BUFFER = 5000
    SORT_BUFFER = 512

    def __init__(self, dataset, percent: float = 1.0):
        self.huggingface_dataSet = dataset.shuffle(seed=42).select(range(int(len(dataset) * percent)))

    def __len__(self):
        return len(self.huggingface_dataSet)

    def __getitem__(self, idx):
        return DirtExample.from_dataline(self.huggingface_dataSet[idx])
