"""Predefined broader probes and scoring; no generation-dependent selection."""
import ast
from collections import Counter
import json
import random
import re
import string
from common import ROOT
from cass.config import DATA_DIR
from cass.tasks import TaskData

NLP_FILES = {
    'news-topic': 'abstractive/ag_news.json',
    'sentiment': 'abstractive/sentiment.json',
    'commonsense-choice': 'abstractive/commonsense_qa.json',
    'reading-comprehension': 'extractive/squad_val.json',
    'entity-person': 'extractive/conll2003_person.json',
    'entity-organization': 'extractive/conll2003_organization.json',
    'entity-location': 'extractive/conll2003_location.json',
}


def make_task(name,pairs,family,seed=20260927,n_eval=100):
    seen=set(); unique=[]
    for x,y in pairs:
        if x not in seen:
            seen.add(x); unique.append((x,y))
    random.Random(seed).shuffle(unique)
    return TaskData(name=name,family=family,eval_queries=unique[:n_eval],
                    fewshot_pool=unique[n_eval:n_eval+40],dict_pool=unique[n_eval+40:])


def load_broader():
    tasks={}
    for name,rel in NLP_FILES.items():
        data=json.loads((DATA_DIR/rel).read_text())
        pairs=[(str(x['input']).strip(),str(x['output']).strip()) for x in data]
        tasks[name]=make_task(name,pairs,'semantic')
    # Long structured output: each input contains 20--30 independently sampled
    # words. The output must preserve order and transform every word.
    words=['river','forest','garden','music','window','table','paper','silver',
           'winter','summer','morning','evening','travel','planet','ocean','cloud',
           'bright','quiet','green','yellow','stone','market','friend','engine',
           'orange','purple','doctor','teacher','valley','mountain','island','bridge',
           'castle','village','flower','cotton','velvet','copper','circle','square']
    rng=random.Random(20260927)
    for kind in ['title','upper','reverse-words']:
        pairs=[]
        for _ in range(180):
            ws=rng.sample(words,rng.randint(20,30))
            x=' '.join(ws)
            y={'title':lambda:' '.join(w.title() for w in ws),
               'upper':lambda:x.upper(),
               'reverse-words':lambda:' '.join(reversed(ws))}[kind]()
            pairs.append((x,y))
        tasks['long-'+kind]=make_task('long-'+kind,pairs,'long-structured',n_eval=60)
    # Single-expression code generation. Scoring parses and evaluates only a
    # fixed whitelist of AST nodes/functions, with no Python exec or eval.
    functions=['sum','max','min','sorted']
    for fn in functions:
        pairs=[]
        for _ in range(180):
            xs=rng.sample(range(1,100),rng.randint(4,8))
            x=str(xs); y=f'{fn}({x})'
            pairs.append((x,y))
        tasks['code-'+fn]=make_task('code-'+fn,pairs,'code-expression',n_eval=60)
    for fn in ['sum','max','min']:
        pairs=[]
        for _ in range(180):
            xs=rng.sample(range(1,100),rng.randint(4,8))
            x=str(xs); y=f'{fn}(sorted({x})[:3])'
            pairs.append((x,y))
        tasks['code-sort3-'+fn]=make_task('code-sort3-'+fn,pairs,'code-multistep',n_eval=60)
    return tasks


def clean_answer(pred):
    pred=pred.strip()
    if pred.startswith('```'):
        lines=pred.splitlines()
        pred='\n'.join(lines[1:]).split('```')[0].strip()
    return pred.split('\nQ:')[0].strip().splitlines()[0] if pred else ''


def normalized(s):
    s=s.lower().translate(str.maketrans('','',string.punctuation))
    s=re.sub(r'\b(a|an|the)\b',' ',s)
    return s.split()


def token_f1(a,b):
    aa,bb=normalized(a),normalized(b)
    overlap=sum((Counter(aa)&Counter(bb)).values())
    return 2*overlap/(len(aa)+len(bb)) if aa or bb else 1.


def safe_expression(source):
    tree=ast.parse(source,mode='eval')
    if sum(1 for _ in ast.walk(tree))>150: raise ValueError('expression too large')
    def walk(n):
        if isinstance(n,ast.Expression): return walk(n.body)
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): return n.value
        if isinstance(n,(ast.List,ast.Tuple)): return [walk(e) for e in n.elts]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub): return -walk(n.operand)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['sum','min','max','sorted']:
            if len(n.args)!=1 or n.keywords: raise ValueError('unsupported call')
            value=walk(n.args[0])
            if not isinstance(value,list): raise ValueError('list required')
            return {'sum':sum,'min':min,'max':max,'sorted':sorted}[n.func.id](value)
        if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Slice):
            value=walk(n.value)
            a=walk(n.slice.lower) if n.slice.lower else None
            b=walk(n.slice.upper) if n.slice.upper else None
            c=walk(n.slice.step) if n.slice.step else None
            return value[slice(a,b,c)]
        raise ValueError('unsupported AST')
    # Require a generated operation; a copied literal answer is not code success.
    if not any(isinstance(n,ast.Call) for n in ast.walk(tree)): raise ValueError('no operation')
    return walk(tree)


def score(name,pred,target):
    p=clean_answer(pred)
    if name.startswith('code-'):
        try: return float(safe_expression(p)==safe_expression(target))
        except (ValueError,TypeError,SyntaxError,IndexError): return 0.
    if name.startswith('long-'):
        return float(' '.join(p.split())==' '.join(target.split()))
    if name in ['reading-comprehension','entity-person','entity-organization','entity-location']:
        return token_f1(p,target)
    if name=='commonsense-choice':
        match=re.match(r'^\(?([a-e])\)?(?:[.:\s]|$)',p.lower())
        return float(bool(match) and match.group(1)==target.lower())
    return float(p.strip(' .:').lower()==target.lower())


if __name__=='__main__':
    assert safe_expression('sum(sorted([8, 1, 5, 3])[:3])')==9
    assert score('code-sum','__import__("os").system("true")','sum([1,2])')==0
    assert score('code-sum','3','sum([1,2])')==0
    assert score('reading-comprehension','The Denver Broncos','Denver Broncos')==1
    tasks=load_broader()
    print({n:dict(family=t.family,n_eval=len(t.eval_queries),n_few=len(t.fewshot_pool)) for n,t in tasks.items()})
