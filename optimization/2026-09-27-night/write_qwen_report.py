"""Report all controlled factors and separately frozen Qwen transfer checks."""
import hashlib
import json
from common import HERE, ROOT


def main():
    first=json.loads((HERE/'qwen_factorial_summary.json').read_text())
    second=json.loads((HERE/'qwen_prompt_factorial_summary.json').read_text())
    live=json.loads((HERE/'live_summary.json').read_text())
    robustness=json.loads((HERE/'metric_robustness.json').read_text())['experiments']
    lines=['# Qwen: intervention and prompt diagnostics', '',
        'Two controlled oracle studies distinguish the injection primitive from '
        'few-shot support estimation. Each evaluates all specified conditions on '
        '20 known targets excluded from development-target selection. Oracle '
        'task activations remain available in these diagnostic studies. They '
        'are not unseen-task steering scores.', '',
        '## Prompt/layer/gain/position factorial', '',
        '| Prompt | Layers | Gain | Prompt positions | Oracle accuracy |',
        '|---|---|---:|---:|---:|']
    for prompt in ['plain','chat']:
        for layers in ['(14, 20)','(24,)']:
            for gain in [1.,2.]:
                for positions in [1,4]:
                    key=f'{prompt}_layers{layers}_gain{gain}_positions{positions}'
                    lines.append(f'| {prompt} | {layers} | {gain:g} | {positions} | {first["configuration_means"][key]:.4f} |')
    lines += ['',
        'The native/chat condition includes a task-agnostic system instruction. '
        'The layer-setting contrast changes both depth and the number of '
        'intervened layers; it does not isolate depth alone. Prompt positions '
        'affect prefill; generation continues with the prescribed final-token '
        'intervention. Gain and the other factors remain fixed in each paired '
        'contrast.', '',
        '## Separate format and instruction', '',
        'All four prompt protocols are remined on the same device with identical '
        '100-pair/10-example mining data and batch-8 extraction boundaries. '
        'Gain is two and one prompt position is intervened. The instruction '
        'is identical in content in plain and native formats:', '',
        '> Infer the input-output rule from the examples when provided. Return only the output, without explanation.', '',
        '| Native format | Generic instruction | Layers 14+20 | Layer 24 |',
        '|---|---|---:|---:|']
    for native in [0,1]:
        for instruction in [0,1]:
            vals=[second['means'][f'native{native}_instruction{instruction}_layers{layers}'] for layers in ['(14, 20)','(24,)']]
            lines.append(f'| {bool(native)} | {bool(instruction)} | {vals[0]:.4f} | {vals[1]:.4f} |')
    lines += ['',
        'At native layer24, the instruction effect is +20.9 percentage points '
        '[12.4, 30.7]. Holding the instruction fixed, native versus plain format '
        'at layer24 differs by +1.6 [-2.6, 5.9]. The evidence identifies an '
        'instruction/intervention interaction; it does not establish native '
        'format alone as the remedy or identify a unique architectural cause.', '',
        'Every conditional paired effect, including unfavorable conditions, is '
        'retained in `qwen_factorial_summary.json` and '
        '`qwen_prompt_factorial_summary.json`. Intervals resample 20 task means '
        'with 50,000 draws; these are individual contrast intervals, not '
        'simultaneous coverage over all factorial contrasts.', '',
        '## Separately frozen few-shot confirmation', '',
        'The initial Qwen3 intervention is selected on the original 12 '
        'development targets. It uses native format plus the generic instruction, '
        'a later-layer setting and the original shuffled contrast. The '
        'confirmation uses the same four examples in every arm and three new '
        'demonstration seeds. These results include support estimation.', '',
        '| Suite | Original plain CASS | Selected plain CASS | Selected native CASS | Native z-only | Native ICL4 |',
        '|---|---:|---:|---:|---:|---:|']
    for suite,label in [('heldout_known','Non-development LOTO20'),('novel','Novel15'),('compound','Compound10')]:
        r=live['qwen_confirm']['suites'][suite]['means']
        methods=['plain_original_cass','plain_selected_cass','chat_selected_cass','chat_z','chat_icl4']
        lines.append('| '+label+' | '+' | '.join(f'{r[m]:.4f}' for m in methods)+' |')
    lines += ['',
        'The oracle is a reference intervention, not an intrinsic accuracy '
        'ceiling. The instruction adds prompt content and is included in both '
        'native CASS and native ICL controls. These Qwen results do not inherit '
        'the Llama-specific timing measurements.', '']
    for model,split in [(m,s) for m in ['qwen3-4b','qwen25-3b'] for s in ['confirm','fresh']]:
        stem=model+'_contrast_'+split
        split_label='original queries' if split=='confirm' else '50 additional queries per task'
        if not (HERE/(stem+'_done.json')).exists() or stem not in live or stem not in robustness:
            lines += [f'{model}, {split_label}: pending complete confirmation.', ''];continue
        selected=json.loads((HERE/(model+'_contrast_selected.json')).read_text())
        lines += [f'### {model}: {split_label}', '',
                  'One setting is selected using the same 12 development tasks and '
                  'then applied to all 45 non-development confirmation targets. '
                  '“Best overall” is the development winner including zero correction; '
                  'it is not selected on confirmation accuracy. Native prompting '
                  'includes the same generic instruction in every native arm. '
                  'The plain-original arm uses the submitted layers 14+20, gain 1 '
                  'configuration. Qwen2.5 plain activations are freshly mined, '
                  'so that arm is a matched rerun of the method rather than a '
                  'bitwise replication of the original checkpointed activations.', '',
                  'The additional-query check keeps the same demonstrations and '
                  'settings, using 50 target inputs disjoint from the target’s '
                  'original evaluation and few-shot inputs. It was specified '
                  'after the Qwen3 original-query result, without retuning, and '
                  'before either model’s additional-query evaluation.', '',
                  '```json',json.dumps({k:v for k,v in selected.items() if k!='all_development_means'},indent=2),'```','',
                  'Submitted prefix metric:', '',
                  '| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        suites=[('heldout_known','Non-development LOTO20',20),('novel','Novel15',15),('compound','Compound10',10)]
        for suite,label,n in suites:
            entry=live[stem]['suites'][suite];assert entry['n_complete_tasks']==n
            r=entry['means']
            methods=['best_correction','matched_no_correction','best_overall','native_shuffle24','plain_original','icl4']
            lines.append('| '+label+' | '+' | '.join(f'{r[m]:.4f}' for m in methods)+' |')
        lines += ['', '| Suite | Paired comparison | Difference, pp [95% CI] |', '|---|---|---:|']
        for suite,label,n in suites:
            for key,v in live[stem]['suites'][suite].items():
                if '__' not in key:continue
                lo,hi=v['ci95']
                lines.append(f"| {label} | {key.replace('__',' − ')} | {100*v['diff']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
        lines += ['', 'Case-aware literal-first-line metric:', '',
                  '| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for suite,label,n in suites:
            entry=robustness[stem][suite]['task_case_literal_line'];assert entry['n_tasks']==n
            lines.append('| '+label+' | '+' | '.join(f"{entry['means'][m]:.4f}" for m in methods)+' |')
        lines += ['', '| Suite | Paired comparison | Difference, pp [95% CI] |', '|---|---|---:|']
        for suite,label,n in suites:
            for key,v in robustness[stem][suite]['task_case_literal_line']['paired'].items():
                lo,hi=v['ci95']
                lines.append(f"| {label} | {key.replace('__',' − ')} | {100*v['diff']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
        lines += ['', 'Generation source SHA256: `'+hashlib.sha256((HERE/(stem+'.jsonl')).read_bytes()).hexdigest()+'`.', '']
    target=ROOT/'rebuttal/2026-09-27-optimized/qwen_evidence.md'
    target.write_text('\n'.join(lines));print(target)


if __name__=='__main__':main()
