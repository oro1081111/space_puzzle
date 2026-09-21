"""Export the selected frozen actor and numerical parity fixtures for the browser."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from sb3_contrib import MaskablePPO
from training.engine import Game, simple_action
from training.search import choose_action


class Actor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        features = self.policy.extract_features(obs)
        return self.policy.action_net(self.policy.mlp_extractor.forward_actor(features))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,choices=[15,30],default=15)
    size=parser.parse_args().size
    suffix='' if size==15 else '-30'
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    source = root / 'training/runs/redesign15/selected_v2.zip'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'ff89c359922f6c2e3292270ee26aeebe1e2609b5ea6e9383b95bd2c1fc6a6db4'
    model = MaskablePPO.load(source, device='cpu')
    actor = Actor(model.policy).eval()
    target = root / f'grid-clash/atlas-r-v1{suffix}.onnx'
    torch.onnx.export(actor, torch.zeros(2, 7, size, size), str(target),
                      input_names=['board'], output_names=['logits'],
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(19091)
    fixtures = []
    for episode in range(4):
        game = Game(size=size)
        for step in range(100):
            obs = np.stack([game.observation(i) for i in range(2)])
            with torch.no_grad():
                expected = actor(torch.from_numpy(obs)).numpy()
            actual = session.run(None, {'board': obs})[0]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            if step % 5 == 0:
                fixtures.append(dict(state=game.state(), logits=expected.tolist(),
                                     actions=[choose_action(model, game, i) for i in range(2)]))
            game.step([simple_action(game, i, rng) for i in range(2)])
            if game.ended:
                break
    (root / f'tests/atlas-fixtures{suffix}.json').write_text(json.dumps(fixtures), encoding='utf-8')
    metadata = dict(name='ATLAS-R｜全域競速', version='1.0', id=f'atlas-r-{size}x{size}-2p-v1.0'+('-experimental' if size==30 else ''),
                    board_size=size, players=2, experimental=size==30, model_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    search=dict(depth=1, expected=.75, worst=.25, prior=.5))
    (root / f'grid-clash/atlas-r-v1{suffix}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported {target.stat().st_size} bytes; {len(fixtures)} parity fixtures')
