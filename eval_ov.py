# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import torch
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from optimum.intel.openvino import OVModelForCausalLM

log: Logger = utils.get_logger("spinquant")


def train() -> None:
    model_args, training_args, ptq_args = process_args_ptq()

    model = OVModelForCausalLM.from_pretrained(
        model_args.input_model
    )
    # if process_word_embeddings:
    #     model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # model.cuda()

    # model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    model.config.max_length = training_args.model_max_length

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B", #model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator_ov(model, testloader, 'cpu', ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))


if __name__ == "__main__":
    train()
