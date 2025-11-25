import json
from utils import (
    build_mamba_and_tokenizer, 
    set_deterministic, 
    parse_options
)
import logging
import sys
import os
# import this will use lm_eval logging format (see lm_eval/logger.py and lm_eval/__main__.py)
from quamba.eval_utils import eval_mamba_few_shot, eval_mamba_generation, evaluate_ppl
from quamba.modelutils_mamba import quantize_model_mamba

def main(args):
    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0] # Assume that the models name is like "model_type-<model_size, model version>"
    assert model_name != None, "Please check the model path."
    logging.info(f"Creating Model:{model_name}")

    # 初始化percentile logger
    from quamba.percentile_logger import reset_percentile_logger
    plogger = reset_percentile_logger(
        log_file="percentileRangeResults/experiments.jsonl",
        save_activations=True  # 启用激活值保存，用于复现实验
    )
    plogger.log_config(args)

    # Setup Quamba mode using unified interface
    from quamba.mode_config import setup_quamba_mode
    setup_quamba_mode(args.mode, verbose=True)

    model, tokenizer, is_quamba = build_mamba_and_tokenizer(args, model_type)
    model.config.use_cache = False
    logs = {}

    # For Mode 5/6, replace the forward method with the dual-path version
    if args.mode in ['5', '6']:
        if not hasattr(model, f'forward_mode{args.mode}'):
            logging.error(f"Model does not support Mode {args.mode}. Please ensure the model has forward_mode{args.mode} method.")
            raise NotImplementedError(f"Mode {args.mode} not implemented for this model")

        logging.info(f"Using Mode {args.mode} dual-path forward method")
        # Store original forward for reference
        model._original_forward = model.forward
        # Replace with mode-specific forward
        if args.mode == '5':
            model.forward = model.forward_mode5
        elif args.mode == '6':
            model.forward = model.forward_mode6

    if args.quantize:
        """
        Start evaluating Quantized Models from here
        """
        if not is_quamba:
            model = quantize_model_mamba(model, model_type, tokenizer, "cuda", args)
    else:
        """
        Evaluate the non-quantized models
        """
        logging.info(f"Evaluating the performance of fp16 model")
    model.eval()
    
    logs = {}
    if args.eval_ppl:
        logging.info(f"Evaluating ppl result (quantized), dataset: {args.ppl_dataset}")
        ppl_results = evaluate_ppl(model, tokenizer, model_name, batch_size=args.batch_size, device="cuda", dataset=args.ppl_dataset)
        logs['ppl'] = ppl_results
    if args.eval_zero_shot:
        logging.info(f"Evaluating result using lm_eval (quantized), task(s): {args.task_list}")
        lm_eval_results = eval_mamba_few_shot(
            model, tokenizer, 
            model_type=model_type,
            task_list=args.task_list, 
            batch_size=args.batch_size,
            limit=100 if args.testing else None
        )
        logs['lm_eval'] = lm_eval_results['results']
    if args.eval_few_shot:
        logging.info(f"Evaluating {args.fewshot}-shot result using lm_eval (quantized), task(s): {args.task_list}")
        lm_eval_results = eval_mamba_few_shot(
            model, tokenizer, 
            model_type=model_type,
            task_list=args.task_list, 
            batch_size=args.batch_size,
            fewshot=args.fewshot,
            limit=100 if args.testing else None
        )
        logs['lm_eval'] = lm_eval_results['results']
    if args.eval_generation:
        logging.info(f"Evaluating generation result using lm_eval (quantized), task(s): {args.task_list}")
        lm_eval_results = eval_mamba_generation(
            model, tokenizer, 
            model_type=model_type,
            task_list=args.task_list, 
            batch_size=args.batch_size,
            fewshot=args.fewshot,
            limit=100 if args.testing else None
        )
        logs['lm_eval'] = lm_eval_results['results']
    if not args.eval_ppl and not args.eval_zero_shot and not args.eval_few_shot and not args.eval_generation:
        logging.warning("No task to run with, try `--eval_ppl`, `--eval_zero_shot`, `--eval_generation`, `--eval_few_shot --fewshot n`?")
        
    if args.log_dir:
        from datetime import datetime

        # Add timestamp to logs
        logs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs['args'] = vars(args)

        os.makedirs(args.log_dir, exist_ok=True)
        if args.quantize:
            log_name = f"{model_name}" if is_quamba else f"{model_name}_w{args.w_bits}a{args.a_bits}"
            log_paths = os.path.join(args.log_dir, f"{log_name}.json")
        else:
            log_paths = os.path.join(args.log_dir, f"{model_name}_fp16.json")
        with open(log_paths, 'a') as fp:
            logging.info(f"Saving result to {log_paths}")
            json.dump(logs, fp, indent=4)
            fp.write('\n')  # Add newline for better separation between runs

    # 记录最终结果到percentile logger
    if 'lm_eval' in logs:
        # 提取accuracy和perplexity
        for task, results in logs['lm_eval'].items():
            if 'acc,none' in results and 'perplexity,none' in results:
                accuracy = results['acc,none']
                perplexity = results['perplexity,none']
                plogger.log_results(accuracy, perplexity)
                break

    # 保存percentile实验日志
    plogger.print_summary()
    plogger.save()

    # Print command line for easy screenshot reference
    print("\n" + "="*80)
    print(f"  python3 {' '.join(sys.argv)}")
    print("="*80 + "\n")

if __name__ =='__main__':    
    set_deterministic(1234)
    args = parse_options()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    main(args)

