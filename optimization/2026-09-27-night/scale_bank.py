"""224 explicit synthetic skills extending the 32 real tasks to 256 skills.

They are a controlled dictionary-size/coherence stress test, not 224 independent
natural-language domains. No generated task equals a held-out novel/compound task.
"""
import json
import random
import string
from common import HERE
from cass.tasks import TaskData


def split(name,pairs,family):
    rng=random.Random(20260927)
    pairs=list(dict(pairs).items());rng.shuffle(pairs)
    assert len(pairs)>=180
    return TaskData(name=name,family=family,eval_queries=pairs[:40],
                    fewshot_pool=pairs[40:70],dict_pool=pairs[70:])


def bank_tasks():
    out={}
    for a in range(2,10):
        for b in range(-8,9):
            name=f'affine_a{a}_b{b}'
            out[name]=split(name,[(str(x),str(a*x+b)) for x in range(-200,201)],'affine-arithmetic')
    for b in range(-2,3):
        name=f'affine_a10_b{b}'
        out[name]=split(name,[(str(x),str(10*x+b)) for x in range(-200,201)],'affine-arithmetic')
    for modulus in range(2,18):
        name=f'modulo_{modulus}'
        out[name]=split(name,[(str(x),str(x%modulus)) for x in range(401)],'modular-arithmetic')
    rng=random.Random(271828)
    words=['river','forest','garden','music','window','table','paper','silver',
           'winter','summer','morning','evening','travel','planet','ocean','cloud',
           'bright','quiet','green','yellow','stone','market','friend','engine',
           'orange','purple','doctor','teacher','valley','mountain','island','bridge']
    for length in range(3,11):
        sequences=[rng.sample(words,length) for _ in range(260)]
        for index in range(length):
            if length==5 and index in [0,2,4]:continue  # already in original dictionary
            name=f'list{length}_position{index+1}'
            out[name]=split(name,[(', '.join(xs),xs[index]) for xs in sequences],'indexed-list-selection')
    strings=[''.join(rng.choices(string.ascii_lowercase,k=rng.randint(12,16))) for _ in range(260)]
    for index in range(1,10):  # first-letter transforms already in original dictionary
        for case in ['lower','upper']:
            name=f'char_position{index+1}_{case}'
            out[name]=split(name,[(x,x[index].upper() if case=='upper' else x[index]) for x in strings],
                            'indexed-character-selection')
    assert len(out)==224,len(out)
    return out


def ordered_bank_names():
    # Stratified round robin gives each scale all four generated families.
    tasks=bank_tasks(); by_family={}
    for name,t in tasks.items(): by_family.setdefault(t.family,[]).append(name)
    rng=random.Random(314159)
    for names in by_family.values(): rng.shuffle(names)
    order=[]
    while any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family]: order.append(by_family[family].pop())
    return order


if __name__=='__main__':
    tasks=bank_tasks()
    manifest=dict(n_original=32,n_synthetic=len(tasks),sizes=[32,64,128,256],
                  order=ordered_bank_names(),
                  families={n:t.family for n,t in tasks.items()},
                  scope='controlled synthetic scaling and coherence stress test')
    (HERE/'scale_bank_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Created manifest for',len(tasks),'synthetic skills')
