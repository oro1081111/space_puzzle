import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from training.engine import Game, RULE_VERSION, simple_action

ROOT = Path(__file__).resolve().parents[1]


class Oracle:
    def __init__(self):
        self.process = subprocess.Popen(['node', str(ROOT/'training/oracle.cjs')],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding='utf-8')

    def call(self, **message):
        self.process.stdin.write(json.dumps(message)+'\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('JavaScript oracle exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()


def verify(games=24, steps=400, size=30):
    oracle = Oracle()
    rng = np.random.default_rng(20260921)
    comparisons = collisions = completed_games = 0
    start = time.perf_counter()

    def compare(game, actual):
        nonlocal comparisons
        expected = game.state()
        for key, value in expected.items():
            assert actual[key] == value, f'{key}: Python/JavaScript mismatch at comparison {comparisons}'
        comparisons += 1

    try:
        for episode in range(games):
            count = 2 + episode % 3
            game = Game(count,size=size)
            compare(game, oracle.call(op='reset', count=count,size=size))
            for _ in range(steps):
                actions = [simple_action(game, i, rng, greedy=True) for i in range(count)]
                advanced = game.step(actions)
                actual = oracle.call(op='step', actions=actions)
                assert advanced == actual['advanced']
                compare(game, actual)
                if game.ended:
                    completed_games += 1
                    break
            collisions += game.collisions

        for _ in range(2):
            game = Game(size=size)
            oracle.call(op='reset', count=2,size=size)
            for decision in range(15000):
                actions = [simple_action(game,i,rng) for i in range(2)]
                advanced = game.step(actions)
                actual = oracle.call(op='step',actions=actions)
                assert advanced == actual['advanced']
                compare(game,actual)
                if game.ended:
                    completed_games += 1
                    break
            assert game.ended, 'Full-game differential check did not finish'

        # Cancellation, forced idle, expiry, permanent reach across a temporary ban.
        for block in (False, True):
            game = Game()
            state = game.state()
            state['board'] = [[-1]*30 for _ in range(30)]
            state['positions'] = [[1,2],[3,2]]
            state['board'][2][1] = 0
            state['board'][2][3] = 1
            state['blocks'] = [[2,2]] if block else []
            game.load(state)
            oracle.call(op='reset', count=2, **state)
            for actions in ([1,3],[0,3],[1,0]):
                advanced = game.step(actions)
                actual = oracle.call(op='step', actions=actions)
                assert advanced == actual['advanced']
                compare(game, actual)
        for wall in (5,14,15):
            game = Game()
            state = game.state()
            state['board'] = [[-1]*30 for _ in range(30)]
            state['positions'] = [[wall,15],[0,15]]
            for row in state['board']:
                row[wall] = 0
            state['board'][15][0] = 1
            game.load(state)
            oracle.call(op='reset', count=2, **state)
            game.capture()
            compare(game, oracle.call(op='capture'))
        game = Game()
        state = game.state()
        state['turn'] = 7200
        game.load(state)
        oracle.call(op='reset',count=2,**state)
        game.step([2,0])
        compare(game,oracle.call(op='step',actions=[2,0]))
        assert game.ended
    finally:
        oracle.close()
    return dict(passed=True, board_size=size,comparisons=comparisons, random_collisions=collisions,completed_games=completed_games,
                seconds=time.perf_counter()-start, rule_version=RULE_VERSION,
                source_sha256=hashlib.sha256((ROOT/'grid-clash/index.html').read_bytes()).hexdigest())


def benchmark(steps=10000,size=30):
    game = Game(size=size)
    rng = np.random.default_rng(11)
    start = time.perf_counter()
    finished = 0
    for _ in range(steps):
        game.step([simple_action(game,i,rng) for i in range(2)])
        if game.ended:
            finished += 1
            game.reset()
    seconds = time.perf_counter()-start
    return dict(decisions=steps, seconds=seconds, decisions_per_second=steps/seconds, completed_games=finished)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--size',type=int,choices=[15,30],default=30)
    args=parser.parse_args()
    result=verify(size=args.size)
    result['benchmark']=benchmark(size=args.size)
    target=Path(args.output or ('training/runs/verification.json' if args.size==30 else f'training/runs/verification{args.size}.json'))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
