# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger
import argparse
import sys
import torch
from transformers import LlamaTokenizerFast
from transformers import AutoModelForCausalLM
from eval_utils.main import ptq_model
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from optimum.intel.openvino import OVModelForCausalLM

from nncf.torch.model_creation import load_from_config
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage

log: Logger = utils.get_logger("spinquant")


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--input_model",
        type=str,
        default="",
        help="The model id or path.",
    )
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="The tokenizer id or path.",
    )
    parser.add_argument("--model_max_length", type=int, default=2048, help="Evaluation data context length.")
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default=None,
        help="Run pytorch checkpoint.",
    )
    
    return parser


def main(argv) -> None:
    parser = get_argument_parser()
    model_args = parser.parse_args(argv)

    if model_args.ckpt_file:
        model = AutoModelForCausalLM.from_pretrained(model_args.input_model, torch_dtype=torch.bfloat16, device_map="cuda")
        ckpt = torch.load(model_args.ckpt_file, weights_only=False, map_location='cpu')
        model = load_from_config(model, ckpt["nncf_config"])
        hook_storage = get_hook_storage(model)
        hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    else:
        model = OVModelForCausalLM.from_pretrained(
            model_args.input_model
        )
    # if process_word_embeddings:
    #     model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # model.cuda()

    # model = ptq_model(ptq_args, model, model_args)
    model.seqlen = model_args.model_max_length
    model.config.max_length = model_args.model_max_length

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.tokenizer_id,
        cache_dir=None,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=None,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    testloader = data_utils.get_wikitext2(
        seed=0,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator_ov(model, testloader, 'cpu')
    log.info("wiki2 ppl is: {}".format(dataset_ppl))


if __name__ == "__main__":
    main(sys.argv[1:])
