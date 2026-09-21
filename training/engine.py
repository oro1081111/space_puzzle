"""Square simultaneous-move engine; permanent reachability ignores entry bans."""
import math
import json
import numpy as np
from scipy.ndimage import label

RULE_VERSION = 'exclusive-reach-endgame-cycle-v3'
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Game:
    def __init__(self, count=2, size=30):
        self.reset(count,size)

    def reset(self, count=None, size=None):
        self.n = size if size is not None else self.n
        self.count = count if count is not None else self.count
        self.board = np.full((self.n, self.n), -1, dtype=np.int8)
        self.positions = []
        for i in range(self.count):
            angle = -math.pi / 2 + math.pi * 2 * i / self.count
            center=(self.n-1)/2
            radius=self.n*.36
            x = math.floor(center + math.cos(angle) * radius + .5)
            y = math.floor(center + math.sin(angle) * radius + .5)
            self.positions.append((x, y))
            self.board[y, x] = i
        self.blocks = set()
        self.turn = 0
        self.ended = False
        self.decisions = 0
        self.collisions = 0
        self.idle_attempts = 0
        self.tail_history = []
        self.refresh()

    def load(self, state):
        self.board = np.array(state['board'], dtype=np.int8)
        self.n = len(self.board)
        self.positions = [tuple(p) for p in state['positions']]
        self.count = len(self.positions)
        self.blocks = {tuple(p) for p in state.get('blocks', [])}
        self.turn = state.get('turn', 0)
        self.ended = state.get('ended', False)
        self.decisions = self.collisions = 0
        self.idle_attempts = state.get('idleAttempts', 0)
        self.tail_history = [entry[:] for entry in state.get('tailHistory', [])]
        self.refresh()

    def refresh(self):
        self.reaches = []
        for i, (x, y) in enumerate(self.positions):
            labels, _ = label((self.board == -1) | (self.board == i))
            self.reaches.append((labels == labels[y, x]) & (labels != 0))
        self.recount()

    def recount(self):
        self.scores = [int(np.count_nonzero(self.board == i)) for i in range(self.count)]
        self.active = [bool(np.any(reach & (self.board == -1))) for reach in self.reaches]

    def capture(self):
        self.refresh()
        visits = np.sum(self.reaches, axis=0)
        vacant = self.board == -1
        for i, reach in enumerate(self.reaches):
            self.board[vacant & (visits == 1) & reach] = i
        # Exclusive fills cannot alter anyone's permanent reachability.
        self.recount()

    def mask(self, player):
        result = np.zeros(4, dtype=bool)
        if self.ended or not self.active[player]:
            return result
        x, y = self.positions[player]
        for d, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy
            result[d] = (0 <= nx < self.n and 0 <= ny < self.n
                         and (nx, ny) not in self.blocks
                         and self.board[ny, nx] in (-1, player))
        return result

    def repeated_endgame_clash(self, collided, progress=False):
        if progress or not 2 <= self.n**2-sum(self.scores) <= 4:
            self.tail_history=[]
            return False
        signature=json.dumps([self.positions,sorted(self.blocks,key=lambda p:(p[1],p[0]))],separators=(',',':'))
        self.tail_history.append([signature,collided])
        self.tail_history=self.tail_history[-96:]
        for period in range(1,min(32,len(self.tail_history)//3)+1):
            cycle=self.tail_history[-period:]
            if any(entry[1] for entry in cycle) and cycle==self.tail_history[-2*period:-period]==self.tail_history[-3*period:-2*period]:
                return True
        return False

    def step(self, actions):
        if self.ended:
            return False
        proposals = {}
        for i, action in enumerate(actions):
            if action is None or action not in range(4) or not self.mask(i)[action]:
                continue
            dx, dy = DIRECTIONS[action]
            x, y = self.positions[i]
            proposals.setdefault((x + dx, y + dy), []).append(i)
        self.decisions += 1
        self.idle_attempts += 1
        contested = [pos for pos, players in proposals.items() if len(players) > 1]
        if contested:
            self.blocks.update(contested)
            self.collisions += 1
            self.ended = sum(self.scores) == self.n**2-1 or self.repeated_endgame_clash(True) or self.idle_attempts >= 2*self.n**2
            return False
        before = sum(self.scores)
        self.turn += 1
        for (x, y), players in proposals.items():
            player = players[0]
            self.positions[player] = (x, y)
            if self.board[y, x] == -1:
                self.board[y, x] = player
        self.capture()
        self.blocks.clear()
        if sum(self.scores) > before:
            self.idle_attempts = 0
        repeated=self.repeated_endgame_clash(False,sum(self.scores)>before)
        self.ended = not any(self.active) or sum(self.scores) == self.n ** 2 or repeated or self.turn > self.n ** 2 * 8 or self.idle_attempts >= 2*self.n**2
        return True

    def observation(self, player):
        data = np.zeros((7, self.n, self.n), dtype=np.float32)
        data[0] = self.board == player
        data[1] = (self.board >= 0) & (self.board != player)
        data[2] = self.board == -1
        for owner, (x, y) in enumerate(self.positions):
            data[3 if owner == player else 4, y, x] = 1
        for x, y in self.blocks:
            data[5, y, x] = 1
        data[6].fill(max(0, 1 - self.turn / (self.n ** 2 * 8 + 1)))
        return data

    def state(self):
        return dict(board=self.board.tolist(), positions=[list(p) for p in self.positions],
                    blocks=[list(p) for p in sorted(self.blocks, key=lambda p:(p[1],p[0]))],
                    turn=self.turn, ended=bool(self.ended), scores=self.scores, active=self.active, idleAttempts=self.idle_attempts,tailHistory=self.tail_history)


def simple_action(game, player, rng, greedy=True):
    legal = np.flatnonzero(game.mask(player))
    if not len(legal):
        return None
    if greedy:
        x, y = game.positions[player]
        new = [d for d in legal if game.board[y+DIRECTIONS[d][1], x+DIRECTIONS[d][0]] == -1]
        if new:
            legal = new
    return int(rng.choice(legal))
