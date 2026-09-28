"""Check actual IDs, avoiding a duplicated BOS in native-template Llama."""
import json
from types import SimpleNamespace
from transformers import AutoTokenizer
from common import HERE,DEV_TASKS
from chat_probe import render,SYSTEM
from cass.config import MODEL_PATHS
from cass.tasks import load_task,zs_prompt,icl_prompt


def main():
    records=[]
    for model in ['llama31-8b','qwen3-4b','qwen25-3b']:
        tok=AutoTokenizer.from_pretrained(MODEL_PATHS[model],local_files_only=True)
        wrapper=SimpleNamespace(tok=tok);n=0;old_different=0
        for name in DEV_TASKS:
            task=load_task(name)
            for x,y in task.eval_queries[:2]:
                for prompt in [zs_prompt(x),icl_prompt(task.fewshot_pool[:4],x)]:
                    native=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
                                                  tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    expected=tok.encode(native,add_special_tokens=False)
                    before=tok.encode(native);actual=tok.encode(render(wrapper,prompt))
                    assert actual==expected,(model,name)
                    old_different+=before!=expected;n+=1
        records.append(dict(model=model,n_checks=n,default_encode_native_different=old_different,
                            corrected_render_all_equal=True))
    (HERE/'native_tokenization_checks.json').write_text(json.dumps(dict(checks=records,
        scope='Before any native-template Llama activations or output generations; Qwen token sequences unchanged.',
        reference='https://huggingface.co/docs/transformers/v4.57.1/chat_templating'),indent=2)+'\n')
    print(records)


if __name__=='__main__':main()
