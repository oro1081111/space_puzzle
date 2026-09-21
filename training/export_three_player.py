"""Export a separately selected three-player actor; never overwrite two-player weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from sb3_contrib import MaskablePPO
from training.engine import Game, simple_action
from training.export_web import Actor
from training.imitate import low_load


def export(source):
    low_load()
    root = Path(__file__).resolve().parents[1]
    source = Path(source).resolve()
    actor = Actor(MaskablePPO.load(source, device='cpu').policy).eval()
    target = root / 'grid-clash/atlas-r-3p-v1.onnx'
    torch.onnx.export(actor, torch.zeros(2, 7, 15, 15), str(target),
                      input_names=['board'], output_names=['logits'], opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    fixtures = []
    rng = np.random.default_rng(910073)
    for _ in range(4):
        game = Game(count=3, size=15)
        for step in range(100):
            expected = []
            for owners in ([0, 1], [2, 2]):
                obs = np.stack([game.observation(i) for i in owners])
                with torch.no_grad():
                    values = actor(torch.from_numpy(obs)).numpy()
                np.testing.assert_allclose(session.run(None, {'board': obs})[0], values, atol=2e-5, rtol=2e-5)
                expected.extend(values[:2 if owners[0] == 0 else 1].tolist())
            if step % 10 == 0:
                fixtures.append(dict(state=game.state(), logits=expected))
            game.step([simple_action(game, i, rng) for i in range(3)])
            if game.ended:
                break
    (root / 'tests/atlas-fixtures-3p.json').write_text(json.dumps(fixtures), encoding='utf-8')
    metadata = dict(name='ATLAS-R Trio', version='1.0', board_size=15, players=3,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    source=str(source.relative_to(root)), training='three-player PPO fine-tuning with frozen opponents')
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Export passed: {len(fixtures)} three-player fixtures, {target.stat().st_size} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    export(parser.parse_args().model)
