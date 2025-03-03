import argparse
import logging
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils import EVAL
from gptqmodel.utils.backend import BACKEND
from compile_qwen import *


def parse_args():
    parser = argparse.ArgumentParser(description="GPTQ Quantization and Evaluation Script")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or identifier of the base model to be quantized."
    )
    parser.add_argument(
        "--quantize_model",
        type=str,
        default="",
        help=(
            "Path to store the quantized model. "
            "If not provided, will use <model_name>-W<wbits>A16-gptq as default."
        )
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=4,
        help="Number of bits for weight quantization. e.g. 4 or 8."
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for GPTQ quantization."
    )
    parser.add_argument(
        "--desc_act",
        action="store_true",
        help="Whether to use descending activation in GPTQ (advanced)."
    )
    parser.add_argument(
        "--sym",
        action="store_true",
        help="Whether to enforce symmetric quantization."
    )
    parser.add_argument(
        "--use_triton",
        action="store_true",
        help="Whether to use Triton-based kernels if available."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the quantization and inference on, e.g. 'cuda:0' or 'cpu'."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Flag to indicate whether to run evaluation after quantization."
    )
    parser.add_argument(
        "--eval_framework",
        type=str,
        choices=["lmeval", "evalplus"],
        default="lmeval",
        help="Choose the framework for evaluation (only if --eval is set)."
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="ARC_CHALLENGE",
        help="Comma-separated tasks to evaluate on if --eval is set."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_result.json",
        help="File to store evaluation results (only if --eval is set)."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to run a compile step after quantization (optional)."
    )

    args = parser.parse_args()

    if not args.quantize_model:
        args.quantize_model = f"{args.model}-W{args.wbits}A16-gptq"

    return args


def get_wikitext2(tokenizer, nsamples=256, seqlen=1024):
    traindata = load_dataset(
        "/data/disk0/Dataset/wikitext",
        "wikitext-2-raw-v1",
        split="train"
    ).filter(lambda x: len(x["text"]) >= seqlen)

    dataset_samples = [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]
    return dataset_samples


def gptqmodel(
    model_path,
    quantize_model,
    wbits,
    group_size=128,
    desc_act=False,
    sym=False,
    use_triton=False,
    device="cuda:0"
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=1024)

    quantize_config = QuantizeConfig(
        bits=wbits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        format='gptq',
    )

    model = GPTQModel.load(model_path, quantize_config, trust_remote_code=True).to(device)
    model.quantize(traindataset, tokenizer=tokenizer, backend=BACKEND.TORCH)

    model.save(quantize_model)
    tokenizer.save_pretrained(quantize_model)

    return model, tokenizer


def eval_model(model, eval_framework, output_file, eval_tasks):
    """
    根据 eval_framework 执行不同的评测逻辑
    """
    if eval_framework == 'lmeval':
        # 这里示例任务写死成 ARC_CHALLENGE，可在此自定义需要评测的任务
        tasks = [getattr(EVAL.LM_EVAL, t.strip()) for t in eval_tasks.split(",")]
        lm_eval_results = GPTQModel.eval(
            model,
            framework=EVAL.LM_EVAL,
            tasks=tasks,
            trust_remote_code=True,
            output_file=output_file
        )
        print(f"lm-eval results saved to {output_file}")
    elif eval_framework == 'evalplus':
        tasks = [getattr(EVAL.EVALPLUS, t.strip()) for t in eval_tasks.split(",")]
        evalplus_results = GPTQModel.eval(
            model,
            framework=EVAL.EVALPLUS,
            tasks=tasks,
            trust_remote_code=True,
            output_file=output_file
        )
        print(f"evalplus results saved to {output_file}")


def compile_model(args, model):
    print("Compiling the model...")
    compile(args, model)
    pass


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    logging.info(f"Loading and quantizing model with arguments: {args}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    logging.info(f"After quantized: {args.quantize_model}")

    for name, param in model.state_dict().items():
        print(f"name: {name}, para: \n {param} param.shape: {param.shape}, param.dtype: {param.dtype}")
        

    # model, tokenizer = gptqmodel(
    #     model_path=args.model,
    #     quantize_model=args.quantize_model,
    #     wbits=args.wbits,
    #     group_size=args.group_size,
    #     desc_act=args.desc_act,
    #     sym=args.sym,
    #     use_triton=args.use_triton,
    #     device=args.device
    # )

    # for name, param in model.state_dict().items():
    #     print(f"name: {name}, para: \n {param} param.shape: {param.shape}, param.dtype: {param.dtype}")
    
    # if args.eval:
    #     logging.info("Running evaluation...")
    #     eval_model(args.quantize_model, args.eval_framework, args.output_file, args.eval_tasks)

    # if args.compile:
    #     compile_model(args, args.quantize_model)


if __name__ == "__main__":
    main()
