"""Sequential, low-load development tournament. Does not start training."""
import hashlib
import argparse
import json
from pathlib import Path
from training.evaluate import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='training/runs/pilot15')
    parser.add_argument('--expected-steps',type=int,default=100352)
    args=parser.parse_args()
    run=Path(args.run)
    progress=json.loads((run/'training_report.json').read_text())
    if not progress.get('completed') or progress['steps']!=args.expected_steps:
        raise RuntimeError('Wait for the requested batch to finish before assessment')
    config=json.loads((run/'config.json').read_text())
    latest=str(progress['steps'])
    output=run/f'assessment_{latest}'
    output.mkdir(exist_ok=True)
    models={
        'untrained':run/'untrained.zip',
        latest:run/'latest.zip',
    }
    if config.get('size',30)==30 and args.expected_steps==301056:
        models={ 'untrained':models['untrained'],'98304':run/'snapshot_000098304.zip',
                '198656':run/'snapshot_000198656.zip',latest:models[latest] }
    hashes={name:hashlib.sha256(file.read_bytes()).hexdigest() for name,file in models.items()}
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['sha256']!=hashes:
        raise RuntimeError('Model hashes changed; use a new assessment directory')
    manifest.write_text(json.dumps(dict(models={k:str(v) for k,v in models.items()},sha256=hashes,
        board_size=config.get('size',30),seeds=list(range(9100,9105)),heuristic_seeds=[9200,9201],training_steps=progress['steps']),indent=2),encoding='utf-8')
    reports={}
    def suite(name,model,opponents,seeds):
        target=output/f'{name}.json'
        # Completed suites can be resumed; no partial game is counted as a result.
        if target.exists():
            report=json.loads(target.read_text(encoding='utf-8'))
        else:
            report=evaluate(model,opponents,seeds)
            target.write_text(json.dumps(report,indent=2),encoding='utf-8')
        reports[name]=report
        print(json.dumps({'suite':name,'summary':report['summary']}),flush=True)
    for name,model in models.items():
        suite(f'{name}_simple',model,['random','greedy'],range(9100,9105))
    if '198656' in models:
        suite('latest_vs_previous',models[latest],[str(models['198656'])],range(9300,9305))
    suite('latest_heuristic',models[latest],['apex','bastion','sweep'],[9200,9201])
    suite('untrained_heuristic',models['untrained'],['apex','bastion','sweep'],[9200,9201])
    (output/'summary.json').write_text(json.dumps({name:report['summary'] for name,report in reports.items()},indent=2),encoding='utf-8')
    print('Assessment complete; no further training started.',flush=True)


if __name__=='__main__':
    main()
