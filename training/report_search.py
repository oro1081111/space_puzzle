"""Summarize paired, seed-clustered comparisons without changing any candidate."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def read_matches(directory):
    rows=[]
    for path in sorted(directory.glob('redesign15_selected_v2_*.json')):
        rows.extend(json.loads(path.read_text())['matches'])
    if not rows:raise ValueError(f'No completed reports in {directory}')
    if any(not row['finished'] for row in rows):raise ValueError('Unfinished matches')
    return rows


def summarize(rows):
    def section(matches):
        counts=sum(r['decision_count'] for r in matches)
        return dict(games=len(matches),wins=sum(r['outcome']==1 for r in matches),
            draws=sum(r['outcome']==0 for r in matches),losses=sum(r['outcome']==-1 for r in matches),
            mean_decision_ms=sum(r['decision_count']*r['decision_mean_ms'] for r in matches)/max(1,counts),
            mean_turns=float(np.mean([r['turns'] for r in matches])),
            turn_cap=sum(r['turns']>=1801 for r in matches))
    return dict(total=section(rows),opponents={p:section([r for r in rows if r['opponent']==p])
                for p in sorted({r['opponent'] for r in rows})})


def paired_difference(first,second):
    a={(r['opponent'],r['seed'],r['seat']):r['outcome'] for r in first}
    b={(r['opponent'],r['seed'],r['seat']):r['outcome'] for r in second}
    if a.keys()!=b.keys():raise ValueError('Unpaired schedules')
    clusters=defaultdict(list)
    for key in a:clusters[key[1]].append((a[key]-b[key])/2)
    differences=np.array([np.mean(clusters[seed]) for seed in sorted(clusters)])
    rng=np.random.default_rng(15151)
    estimates=differences[rng.integers(len(differences),size=(10000,len(differences)))].mean(axis=1)
    return dict(seed_clusters=len(differences),score_rate_difference=float(differences.mean()),
        bootstrap_95_interval=np.quantile(estimates,[.025,.975]).tolist(),
        note='Paired bootstrap over seeds, keeping both seats and profiles together. Empirical uncertainty only; not unseen-map or human strength.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('training/runs/dagger15'))
    parser.add_argument('--prefix',choices=['final','opening'],default='final')
    args=parser.parse_args()
    matches={name:read_matches(args.root/f'{args.prefix}_{name}') for name in ['policy','race','hybrid']}
    report=dict(methods={name:summarize(rows) for name,rows in matches.items()},
        hybrid_minus_policy=paired_difference(matches['hybrid'],matches['policy']),
        race_minus_policy=paired_difference(matches['race'],matches['policy']),
        hybrid_minus_race=paired_difference(matches['hybrid'],matches['race']))
    report['code_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                           for name in ['search.py','engine.py','evaluate.py','oracle.cjs']}
    (args.root/f'{args.prefix}_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
