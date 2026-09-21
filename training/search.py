"""Bounded simultaneous-move lookahead using public state and a frozen policy."""
from collections import deque
from copy import deepcopy
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4)
def neighbors(size):
    return [[ny*size+nx for nx,ny in ((x,y-1),(x+1,y),(x,y+1),(x-1,y))
             if 0<=nx<size and 0<=ny<size] for y in range(size) for x in range(size)]


def race_value(game,player):
    """Territory now plus a shortest-path race estimate, not a changed game reward."""
    score=game.scores[player]-game.scores[1-player]
    if game.ended:return float(score)
    board=game.board.ravel().tolist(); adjacent=neighbors(game.n); distances=[]
    for owner in range(2):
        x,y=game.positions[owner]; start=y*game.n+x
        distance=[10**6]*len(board);distance[start]=0;queue=deque([start])
        while queue:
            cell=queue.popleft()
            for target in adjacent[cell]:
                if distance[target]==10**6 and board[target] in (-1,owner):
                    distance[target]=distance[cell]+1;queue.append(target)
        distances.append(distance)
    ours=distances[player]; theirs=distances[1-player]
    return float(score+sum((ours[i]<theirs[i])-(ours[i]>theirs[i])
                           for i,value in enumerate(board) if value==-1))


def policy_probabilities(model,game,player):
    mask=game.mask(player)
    if not mask.any():return np.zeros(4)
    if model is None:return mask.astype(np.float64)/mask.sum()
    import torch
    with torch.no_grad():
        dist=model.policy.get_distribution(torch.as_tensor(game.observation(player)[None]),
                                           action_masks=mask[None])
    return dist.distribution.probs[0].cpu().numpy()


def choose_action(model,game,player,depth=1):
    if model is None and depth!=1:raise ValueError('Network-free baseline supports exactly one turn')
    legal=np.flatnonzero(game.mask(player))
    if not len(legal):return None
    if len(legal)==1:return int(legal[0])
    prior=policy_probabilities(model,game,player)
    enemy=1-player; replies=np.flatnonzero(game.mask(enemy)).tolist() or [None]
    enemy_probs=policy_probabilities(model,game,enemy)
    weights=np.array([enemy_probs[b] if b is not None else 1. for b in replies])
    weights/=weights.sum()
    branches=[]
    for action in legal:
        for reply in replies:
            simulated=deepcopy(game); actions=[None,None]
            actions[player]=int(action);actions[enemy]=reply
            simulated.step(actions);branches.append(simulated)
    # ponytail: enumerate the first simultaneous turn, then batch policy rollouts;
    # this is bounded lookahead, not exhaustive minimax or a worst-case guarantee.
    for _ in range(depth-1):
        observations=[];masks=[];targets=[]
        for i,simulated in enumerate(branches):
            if simulated.ended:continue
            for owner in range(2):
                mask=simulated.mask(owner)
                if mask.any():
                    targets.append((i,owner));masks.append(mask)
                    observations.append(simulated.observation(owner))
        if not targets:break
        actions,_=model.predict(np.array(observations),action_masks=np.array(masks),deterministic=True)
        proposals=[[None,None] for _ in branches]
        for (i,owner),action in zip(targets,actions):proposals[i][owner]=int(action)
        for simulated,proposal in zip(branches,proposals):
            if not simulated.ended:simulated.step(proposal)
    values=np.array([race_value(state,player) for state in branches]).reshape(len(legal),len(replies))
    # A small prior breaks close geometric ties; opponent intent remains uncertain.
    scores=.75*(values@weights)+.25*values.min(axis=1)+.5*np.log(np.maximum(prior[legal],1e-8))
    return int(legal[np.argmax(scores)])
