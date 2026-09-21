from collections import OrderedDict
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv
from training.engine import Game, simple_action


class ParallelGame(ParallelEnv):
    metadata = {'name':'grid_clash_v1', 'render_modes':[]}
    possible_agents = ['p0','p1']

    def __init__(self, size=30):
        self.game=Game(size=size)
        self.observation_spaces={p:spaces.Dict({
            'observation':spaces.Box(0,1,(7,size,size),np.float32),
            'action_mask':spaces.MultiBinary(4)}) for p in self.possible_agents}
        self.action_spaces={p:spaces.Discrete(4) for p in self.possible_agents}
        self.agents=[]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observations(self):
        return {p:dict(observation=self.game.observation(i), action_mask=self.game.mask(i).astype(np.int8))
                for i,p in enumerate(self.possible_agents) if p in self.agents}

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.agents=self.possible_agents[:]
        return self.observations(),{p:{} for p in self.agents}

    def step(self, actions):
        players=self.agents[:]
        if not players:
            return {},{},{},{},{}
        self.game.step([actions.get(p) for p in self.possible_agents])
        result=int(np.sign(self.game.scores[0]-self.game.scores[1])) if self.game.ended else 0
        obs=self.observations()
        rewards={'p0':float(result),'p1':float(-result)}
        terminated={p:self.game.ended for p in players}
        infos={p:{} for p in players}
        if self.game.ended:
            self.agents=[]
        return obs,rewards,terminated,{p:False for p in players},infos


class TrainingEnv(gym.Env):
    """One learning seat; opponents frozen for each episode. Rules are symmetric AI rules."""
    def __init__(self, pool=None, shaping=.1, max_decisions=None, size=30):
        self.observation_space=spaces.Box(0,1,(7,size,size),np.float32)
        self.action_space=spaces.Discrete(4)
        self.game=Game(size=size)
        self.pool=Path(pool) if pool else None
        self.shaping=shaping
        self.max_decisions=max_decisions if max_decisions is not None else size*size*12
        self.cache=OrderedDict()
        self.opponent=None
        self.opponent_name='greedy'

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.player=int(self.np_random.integers(2))
        self.opponent=None
        candidates=sorted(self.pool.glob('snapshot_*.zip')) if self.pool else []
        draw=self.np_random.random()
        if candidates and draw < .5:
            # Keep both recent and old opponents. Weights never change during a game.
            file=str(self.np_random.choice(candidates))
            if file not in self.cache:
                from sb3_contrib import MaskablePPO
                self.cache[file]=MaskablePPO.load(file,device='cpu')
                if len(self.cache)>4:
                    self.cache.popitem(last=False)
            self.opponent=self.cache[file]
            self.opponent_name=Path(file).stem
        else:
            self.opponent_name='random' if draw>.9 else 'greedy'
        return self.game.observation(self.player),{}

    def action_masks(self):
        return self.game.mask(self.player)

    def opponent_action(self):
        player=1-self.player
        mask=self.game.mask(player)
        if not mask.any():
            return None
        if self.opponent is not None:
            action,_=self.opponent.predict(self.game.observation(player),action_masks=mask,deterministic=False)
            return int(action)
        return simple_action(self.game,player,self.np_random,self.opponent_name!='random')

    def step(self, action):
        player=self.player
        before=self.game.scores[player]-self.game.scores[1-player]
        actions=[None,None]
        actions[player]=int(action)
        actions[1-player]=self.opponent_action()
        self.advance(actions)
        # No-choice turns are executed by the environment, not learned as chosen actions.
        while not self.game.ended and not self.action_masks().any() and self.game.decisions<self.max_decisions:
            actions=[None,None]
            actions[1-player]=self.opponent_action()
            self.advance(actions)
        difference=self.game.scores[player]-self.game.scores[1-player]
        reward=self.shaping*(difference-before)/(self.game.n**2)
        if self.game.ended:
            reward+=float(np.sign(difference))
        truncated=not self.game.ended and self.game.decisions>=self.max_decisions
        info={}
        if self.game.ended or truncated:
            info=dict(outcome=int(np.sign(difference)) if self.game.ended else None,
                      scores=self.game.scores, decisions=self.game.decisions, turns=self.game.turn,
                      collisions=self.game.collisions, opponent=self.opponent_name, player=player)
        return self.game.observation(player),float(reward),self.game.ended,truncated,info

    def advance(self, actions):
        self.game.step(actions)


class ExpertTrainingEnv(TrainingEnv):
    """Real JS heuristic, with its full planning memory retained for each match."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.oracle=None

    def reset(self, seed=None, options=None):
        gym.Env.reset(self,seed=seed)
        if self.oracle is None:
            from training.verify import Oracle
            self.oracle=Oracle()
        self.player=int(self.np_random.integers(2))
        self.opponent_name=str(self.np_random.choice(['apex','bastion','sweep','greedy']))
        self.game.load(self.oracle.call(op='reset',size=self.game.n,count=2,
            seed=int(self.np_random.integers(2**31)),styles=[self.opponent_name]*2))
        return self.game.observation(self.player),{}

    def opponent_action(self):
        if self.opponent_name=='greedy':
            return simple_action(self.game,1-self.player,self.np_random,True)
        return 'ai'

    def advance(self, actions):
        decisions=self.game.decisions; collisions=self.game.collisions
        actual=self.oracle.call(op='play',actions=actions)
        self.game.load(actual)
        self.game.decisions=decisions+1
        self.game.collisions=collisions+int(not actual['advanced'])

    def close(self):
        if self.oracle:
            self.oracle.close(); self.oracle=None
