from evaluation import CounterFactEval

def init(data, tokenizer):
    global eval_method
    eval_method = CounterFactEval(data, tokenizer)
