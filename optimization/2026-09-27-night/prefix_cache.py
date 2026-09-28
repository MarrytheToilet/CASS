"""Greedy ICL with one reusable demonstration-prefix KV cache per task."""
import copy
import torch


class PrefixICL:
    @torch.no_grad()
    def __init__(self,hlm,examples,validation_prompts,prefix_override=None):
        self.hlm=hlm
        prefix=('\n\n'.join(f'Q: {x}\nA: {y}' for x,y in examples)+'\n\n'
                if prefix_override is None else prefix_override)
        ids=hlm.tok.encode(prefix)
        full=[hlm.tok.encode(p) for p in validation_prompts]
        # Token boundary merges may require dropping the final prefix token.
        while ids and any(row[:len(ids)]!=ids for row in full):ids=ids[:-1]
        assert ids
        self.ids=ids;self.length=len(ids)
        tensor=torch.tensor([ids],device=hlm.device)
        self.cache=hlm.model.model(input_ids=tensor,use_cache=True).past_key_values
        self.cache_bytes=sum(layer.keys.numel()*layer.keys.element_size()+layer.values.numel()*layer.values.element_size() for layer in self.cache.layers)

    @torch.no_grad()
    def generate(self,prompts,max_new_tokens=8,ops=None,layers=None,schedule='all',stop_after_answer=False):
        hlm=self.hlm;batch=len(prompts);ids=[hlm.tok.encode(p) for p in prompts]
        assert all(row[:self.length]==self.ids for row in ids)
        suffix=[row[self.length:] for row in ids];width=max(map(len,suffix));pad=hlm.tok.pad_token_id
        input_ids=torch.tensor([[pad]*(width-len(row))+row for row in suffix],device=hlm.device)
        smask=torch.tensor([[0]*(width-len(row))+[1]*len(row) for row in suffix],device=hlm.device)
        mask=torch.cat([torch.ones((batch,self.length),device=hlm.device,dtype=smask.dtype),smask],dim=1)
        cache=copy.deepcopy(self.cache);cache.batch_repeat_interleave(batch)
        positions=mask.long().cumsum(-1)-1;positions.masked_fill_(mask==0,1);positions=positions[:,-width:]
        eos=hlm.model.generation_config.eos_token_id
        eos=torch.tensor(eos if isinstance(eos,list) else [eos],device=hlm.device)
        alive=torch.ones(batch,device=hlm.device,dtype=torch.bool);generated=[]
        if stop_after_answer:
            from answer_stop import AnswerStop
            stopping=AnswerStop(hlm.tok,0)
        handles=[]
        for operation,layer in zip(ops or [],layers or []):
            def hook(module,inputs,output,op=operation):
                h=output[0] if isinstance(output,tuple) else output
                h[:,-1]=op(h[:,-1]).to(h.dtype)
            handles.append(hlm.layers[layer-1].register_forward_hook(hook))
        try:
            assert schedule in ['all','prefill']
            for step in range(max_new_tokens):
                out=hlm.model(input_ids=input_ids,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True,logits_to_keep=1)
                if step==0 and schedule=='prefill':
                    for handle in handles:handle.remove()
                    handles=[]
                next_ids=out.logits[:,-1].argmax(-1);next_ids=torch.where(alive,next_ids,pad)
                generated.append(next_ids);alive=alive&~torch.isin(next_ids,eos)
                if stop_after_answer:alive=alive&~stopping(torch.stack(generated,dim=1),None)
                cache=out.past_key_values
                if not alive.any():break
                input_ids=next_ids[:,None];mask=torch.cat([mask,torch.ones((batch,1),device=mask.device,dtype=mask.dtype)],dim=1)
                positions=mask.long().sum(-1,keepdim=True)-1
        finally:
            for handle in handles:handle.remove()
        return hlm.tok.batch_decode(torch.stack(generated,dim=1),skip_special_tokens=True)
