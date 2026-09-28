"""Stop only once the broader benchmark's parsed first answer is irrevocable."""
import torch
from transformers import StoppingCriteria


def answer_complete(text):
    s=text.lstrip()
    if s.startswith('```'):
        if '\n' not in s:return False
        body=s.split('\n',1)[1].lstrip()
        return bool(body) and ('\n' in body or '```' in body)
    return '\n' in s


class AnswerStop(StoppingCriteria):
    def __init__(self,tokenizer,start):self.tokenizer=tokenizer;self.start=start
    def __call__(self,input_ids,scores,**kwargs):
        texts=self.tokenizer.batch_decode(input_ids[:,self.start:],skip_special_tokens=True)
        return torch.tensor([answer_complete(t) for t in texts],device=input_ids.device,dtype=torch.bool)
