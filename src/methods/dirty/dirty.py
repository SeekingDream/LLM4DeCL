from src.utils.utils import common_load_dataset
from src.methods.abstract_method import AbstractMethod
from torch.nn.utils.rnn import pad_sequence
from src.utils.DirtDataset import DirtDataset
from src.methods.dirty.utils import util
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union
from src.utils.variable import Stack, Register
from src.utils.DirtDataset import DirtExample
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data.dataloader import DataLoader
from src.utils.dire_types import TypeLibCodec
import _jsonnet
from src.methods.dirty.model.model import TypeReconstructionModel
import pytorch_lightning as pl
from src.methods.dirty.utils.vocab import Vocab
import json
import torch


class DirtyMethod(AbstractMethod):
    def __init__(self, train_set, dev_set, test_set, config, dirty_config, extra_config=None, ckpt=None):
        super().__init__(train_set, dev_set, test_set, config)
        dirty_config = json.loads(_jsonnet.evaluate_file(dirty_config))
        self.dirty_config = dirty_config

        if extra_config:
            extra_config = json.loads(extra_config)
            dirty_config = util.update(dirty_config, extra_config)

        batch_size = dirty_config["train"]["batch_size"]
        train_set = self.train_data
        dev_set = self.val_data

        self.collator = self.DirtCollator(dirty_config["data"])

        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            collate_fn=lambda b: self.collator,
            num_workers=16,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            dev_set,
            batch_size=batch_size,
            collate_fn=lambda b: self.collator,
            num_workers=8,
            pin_memory=True,
        )

        wandb_logger = WandbLogger(name=config['exp_name'], project="dire", log_model=True)
        wandb_logger.log_hyperparams(config)
        # model
        self.model = TypeReconstructionModel(dirty_config)
        self.ckpt = ckpt

        resume_from_checkpoint = ckpt
        if resume_from_checkpoint == "":
            resume_from_checkpoint = None
        self.trainer = pl.Trainer(
            max_epochs=dirty_config["train"]["max_epoch"],
            logger=wandb_logger,
            gpus=1 if config['cuda'] else None,
            auto_select_gpus=True,
            gradient_clip_val=1,
            callbacks=[
                EarlyStopping(
                    monitor="val_retype_acc"
                    if dirty_config["data"]["retype"]
                    else "val_rename_acc",
                    mode="max",
                    patience=dirty_config["train"]["patience"],
                )
            ],
            check_val_every_n_epoch=dirty_config["train"]["check_val_every_n_epoch"],
            progress_bar_refresh_rate=10,
            accumulate_grad_batches=dirty_config["train"]["grad_accum_step"],
            resume_from_checkpoint=resume_from_checkpoint,
            limit_test_batches=dirty_config["test"]["limit"] if "limit" in dirty_config["test"] else 1.0
        )

    class DirtCollator:
        def _annotate(self, example: DirtExample):
            src_bpe_model = self.vocab.source_tokens.subtoken_model
            snippet = example.code_tokens
            snippet = " ".join(snippet)
            sub_tokens = (
                    ["<s>"]
                    + src_bpe_model.encode_as_pieces(snippet)[: self.max_src_tokens_len]
                    + ["</s>"]
            )
            sub_token_ids = (
                    [src_bpe_model.bos_id()]
                    + src_bpe_model.encode_as_ids(snippet)[: self.max_src_tokens_len]
                    + [src_bpe_model.eos_id()]
            )
            setattr(example, "sub_tokens", sub_tokens)
            setattr(example, "sub_token_ids", sub_token_ids)
            setattr(example, "source_seq_length", len(sub_tokens))

            types_model = self.vocab.types
            subtypes_model = self.vocab.subtypes
            src_var_names = []
            tgt_var_names = []
            src_var_types_id = []
            src_var_types_str = []
            tgt_var_types_id = []
            tgt_var_types_str = []
            tgt_var_subtypes = []
            tgt_var_type_sizes = []
            tgt_var_type_objs = []
            tgt_var_src_mems = []
            tgt_names = []
            # variables on registers first, followed by those on stack
            locs = sorted(
                example.source,
                key=lambda x: sub_tokens.index(f"@@{example.source[x].name}@@")
                if f"@@{example.source[x].name}@@" in sub_tokens
                else self.max_src_tokens_len,
            )
            stack_pos = [x.offset for x in example.source if isinstance(x, Stack)]
            stack_start_pos = max(stack_pos) if stack_pos else None
            for loc in locs[: self.max_num_var]:
                src_var = example.source[loc]
                tgt_var = example.target[loc]
                src_var_names.append(f"@@{src_var.name}@@")
                tgt_var_names.append(f"@@{tgt_var.name}@@")
                src_var_types_id.append(types_model.lookup_decomp(str(src_var.typ)))
                src_var_types_str.append(str(src_var.typ))
                tgt_var_types_id.append(types_model[str(tgt_var.typ)])
                tgt_var_types_str.append(str(tgt_var.typ))
                if types_model[str(tgt_var.typ)] == types_model.unk_id:
                    subtypes = [subtypes_model.unk_id, subtypes_model["<eot>"]]
                else:
                    subtypes = [subtypes_model[subtyp] for subtyp in tgt_var.typ.tokenize()]
                tgt_var_type_sizes.append(len(subtypes))
                tgt_var_subtypes += subtypes
                tgt_var_type_objs.append(tgt_var.typ)

                # Memory
                # 0: absolute location of the variable in the function, e.g.,
                #   for registers: Reg 56
                #   for stack: relative position to the first variable
                # 1: size of the type
                # 2, 3, ...: start offset of fields in the type
                def var_loc_in_func(loc):
                    # TODO: fix the magic number for computing vocabulary idx
                    if isinstance(loc, Register):
                        return 1030 + self.vocab.regs[loc.name]
                    else:
                        from src.methods.dirty.utils.vocab import VocabEntry

                        return (
                            3 + stack_start_pos - loc.offset
                            if stack_start_pos - loc.offset < VocabEntry.MAX_STACK_SIZE
                            else 2
                        )

                tgt_var_src_mems.append(
                    [var_loc_in_func(loc)]
                    + types_model.encode_memory(
                        (src_var.typ.size,) + src_var.typ.start_offsets()
                    )
                )
                tgt_names.append(tgt_var.name)

            setattr(example, "src_var_names", src_var_names)
            setattr(example, "tgt_var_names", tgt_var_names)
            if self.rename:
                setattr(
                    example,
                    "tgt_var_name_ids",
                    [self.vocab.names[n[2:-2]] for n in tgt_var_names],
                )
            setattr(example, "src_var_types", src_var_types_id)
            setattr(example, "src_var_types_str", src_var_types_str)
            setattr(example, "tgt_var_types", tgt_var_types_id)
            setattr(example, "tgt_var_types_str", tgt_var_types_str)
            setattr(example, "tgt_var_subtypes", tgt_var_subtypes)
            setattr(example, "tgt_var_type_sizes", tgt_var_type_sizes)
            setattr(example, "tgt_var_src_mems", tgt_var_src_mems)

            return example

        def __init__(self, config):
            self.vocab = Vocab.load(config["vocab_file"])
            with open(config["typelib_file"]) as type_f:
                self.typelib = TypeLibCodec.decode(type_f.read())
            self.max_src_tokens_len = config["max_src_tokens_len"]
            self.max_num_var = config["max_num_var"]
            self.rename = config.get("rename", False)

        def __call__(self, batch) -> DirtExample:
            return self.collate_fn(batch)

        def collate_fn(self,
                       examples: List[DirtExample],
                       ) -> Tuple[
            Dict[str, Union[torch.Tensor, int]], Dict[str, Union[torch.Tensor, List]]
        ]:
            examples = list(map(self._annotate, examples))
            token_ids = [torch.tensor(e.sub_token_ids) for e in examples]
            input = pad_sequence(token_ids, batch_first=True)
            max_time_step = input.shape[1]
            # corresponding var_id of each token in sub_tokens
            variable_mention_to_variable_id = torch.zeros(
                len(examples), max_time_step, dtype=torch.long
            )
            # if each token in sub_tokens is a variable
            variable_mention_mask = torch.zeros(len(examples), max_time_step)
            # the number of mentioned times for each var_id
            variable_mention_num = torch.zeros(
                len(examples), max(len(e.src_var_names) for e in examples)
            )

            for e_id, example in enumerate(examples):
                var_name_to_id = {name: i for i, name in enumerate(example.src_var_names)}
                for i, sub_token in enumerate(example.sub_tokens):
                    if sub_token in example.src_var_names:
                        var_id = var_name_to_id[sub_token]
                        variable_mention_to_variable_id[e_id, i] = var_id
                        variable_mention_mask[e_id, i] = 1.0
                        variable_mention_num[e_id, var_name_to_id[sub_token]] += 1
            # if mentioned for each var_id
            variable_encoding_mask = (variable_mention_num > 0).float()

            src_type_ids = [
                torch.tensor(e.src_var_types, dtype=torch.long) for e in examples
            ]
            src_type_id = pad_sequence(src_type_ids, batch_first=True)
            type_ids = [torch.tensor(e.tgt_var_types, dtype=torch.long) for e in examples]
            target_type_id = pad_sequence(type_ids, batch_first=True)
            assert target_type_id.shape == variable_mention_num.shape

            subtype_ids = [
                torch.tensor(e.tgt_var_subtypes, dtype=torch.long) for e in examples
            ]
            target_subtype_id = pad_sequence(subtype_ids, batch_first=True)
            type_sizes = [
                torch.tensor(e.tgt_var_type_sizes, dtype=torch.long) for e in examples
            ]
            target_type_sizes = pad_sequence(type_sizes, batch_first=True)

            target_mask = src_type_id > 0
            target_type_src_mems = [
                torch.tensor(mems, dtype=torch.long)
                for e in examples
                for mems in e.tgt_var_src_mems
            ]
            target_type_src_mems = pad_sequence(target_type_src_mems, batch_first=True)
            target_type_src_mems_unflattened = torch.zeros(
                *target_mask.shape, target_type_src_mems.size(-1), dtype=torch.long
            )
            target_type_src_mems_unflattened[target_mask] = target_type_src_mems
            target_type_src_mems = target_type_src_mems_unflattened

            # renaming task
            if hasattr(examples[0], "tgt_var_name_ids"):
                name_ids = [
                    torch.tensor(e.tgt_var_name_ids, dtype=torch.long) for e in examples
                ]
                target_name_id = pad_sequence(name_ids, batch_first=True)
            else:
                target_name_id = None

            return (
                dict(
                    index=sum(
                        [
                            [(e.binary, e.name, name) for name in e.src_var_names]
                            for e in examples
                        ],
                        [],
                    ),
                    src_code_tokens=input,
                    variable_mention_to_variable_id=variable_mention_to_variable_id,
                    variable_mention_mask=variable_mention_mask,
                    variable_mention_num=variable_mention_num,
                    variable_encoding_mask=variable_encoding_mask,
                    target_type_src_mems=target_type_src_mems,
                    src_type_id=src_type_id,
                    target_mask=target_mask,
                    target_submask=target_subtype_id > 0,
                    target_type_sizes=target_type_sizes,
                ),
                dict(
                    tgt_var_names=sum([e.tgt_var_names for e in examples], []),
                    target_type_id=target_type_id,
                    target_name_id=target_name_id,
                    target_subtype_id=target_subtype_id,
                    target_mask=target_mask,
                    test_meta=[e.test_meta for e in examples],
                ),
            )

    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def evaluate(self):
        test_set = self.test_data
        test_loader = DataLoader(
            test_set,
            batch_size=self.dirty_config["test"]["batch_size"],
            collate_fn=self.collator,
            num_workers=8,
            pin_memory=True,
        )
        ret = self.trainer.test(self.model, test_dataloaders=test_loader, ckpt_path=self.ckpt)
        json.dump(ret[0], open("test_result.json", "w"))
