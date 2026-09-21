"""Full matches using the actual page controller, ONNX actor, and Python rule parity."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from training.engine import Game, simple_action
from training.verify import Oracle


def run(size, seed):
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(f'grid-clash/atlas-r-v1{"-30" if size==30 else ""}.onnx', options)
    oracle=Oracle(); game=Game(size=size); rng=np.random.default_rng(seed)
    max_idle=escapes=0
    try:
        game.load(oracle.call(op='reset',size=size,count=2,seed=seed,styles=['atlas','atlas']))
        for _ in range(0 if seed==0 else 4):
            game.load(oracle.call(op='step',actions=[simple_action(game,i,rng) for i in range(2)]))
        for attempt in range(size*size*16):
            obs=np.stack([game.observation(i) for i in range(2)])
            logits=session.run(None,{'board':obs})[0].tolist()
            actual=oracle.call(op='atlas',logits=logits)
            game.step(actual['actions'])
            for key,value in game.state().items():
                assert actual[key]==value,(size,seed,attempt,key)
            max_idle=max(max_idle,game.idle_attempts)
            escapes+=any(actual['escape'])
            if game.ended:
                assert sum(game.scores)==size*size, ('Self-play must finish by filling, not a safety cap', size,seed,actual['endReason'])
                return dict(size=size,seed=seed,turn=game.turn,attempts=attempt+1,
                            scores=game.scores,max_idle=max_idle,escape_steps=escapes,reason=actual['endReason'])
        raise AssertionError('Unfinished self-play')
    finally:oracle.close()


if __name__=='__main__':
    results=[]
    for size in (15,30):
        for seed in range(6):
            result=run(size,seed);results.append(result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    Path('training/runs/atlas11-selfplay.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
