import unittest
import numpy as np
import torch
from training.engine import Game, DIRECTIONS
from training.imitate import augment
from training.verify import Oracle
from training.env import ExpertTrainingEnv


class LearningTests(unittest.TestCase):
    def test_search_preserves_public_state_and_respects_bans(self):
        from training.search import race_value,choose_action
        from sb3_contrib import MaskablePPO
        from training.model import HeadAwareCNN
        from training.env import TrainingEnv
        torch.set_num_threads(1)
        model=MaskablePPO('CnnPolicy',TrainingEnv(size=15),n_steps=8,batch_size=8,
            policy_kwargs=dict(features_extractor_class=HeadAwareCNN,normalize_images=False),device='cpu')
        game=Game(size=15); x,y=game.positions[0];game.blocks.add((x,y+1))
        original=game.state()
        self.assertEqual(race_value(game,0),-race_value(game,1))
        for depth in (1,3):
            action=choose_action(model,game,0,depth)
            self.assertTrue(game.mask(0)[action]);self.assertEqual(game.state(),original)
        game.ended=True
        self.assertEqual(race_value(game,0),game.scores[0]-game.scores[1])

    def test_advice_has_no_effect_on_game_or_randomness(self):
        query=Oracle(); control=Oracle()
        try:
            for seed in range(3):
                options=dict(op='reset',size=15,count=2,seed=seed,styles=['apex','sweep'])
                query.call(**options); state=control.call(**options)
                for _ in range(80):
                    game=Game(size=15); game.load(state)
                    for player in range(2):
                        advice=query.call(op='advise',player=player,style='apex')
                        self.assertEqual(advice,query.call(op='advise',player=player,style='apex'))
                        if advice['action'] is not None:
                            self.assertTrue(game.mask(player)[advice['action']])
                    actual=query.call(op='play',actions=['ai','ai'])
                    state=control.call(op='play',actions=['ai','ai'])
                    self.assertEqual(actual,state)
                    if state['ended']:break
        finally:
            query.close(); control.close()

    def test_symmetry_labels_and_masks(self):
        rng=np.random.default_rng(7)
        for action,(dx,dy) in enumerate(DIRECTIONS):
            for _ in range(32):
                obs=torch.zeros(1,7,15,15)
                obs[0,3,7,7]=1; obs[0,0,7+dy,7+dx]=1
                mask=torch.zeros(1,4,dtype=torch.bool); mask[0,action]=True
                rotated,masked,labels=augment(obs,mask,torch.tensor([action]),rng)
                label=int(labels[0]); x,y=DIRECTIONS[label]
                self.assertEqual(rotated[0,0,7+y,7+x],1)
                self.assertTrue(masked[0,label]); self.assertEqual(masked.sum(),1)

    def test_teacher_actions_reproduce_engine(self):
        oracle=Oracle()
        try:
            for seed in range(5):
                game=Game(size=15)
                game.load(oracle.call(op='reset',size=15,count=2,seed=seed,styles=['apex','sweep']))
                for _ in range(100):
                    actual=oracle.call(op='play',actions=['ai','ai'])
                    game.step(actual['actions'])
                    state=game.state()
                    for key in state:
                        self.assertEqual(state[key],actual[key],key)
                    if game.ended:
                        break
        finally:
            oracle.close()

    def test_expert_episode(self):
        env=ExpertTrainingEnv(size=15)
        try:
            obs,_=env.reset(seed=7)
            for step in range(3000):
                legal=np.flatnonzero(env.action_masks())
                self.assertGreater(len(legal),0)
                obs,reward,done,truncated,info=env.step(int(legal[step%len(legal)]))
                self.assertTrue(env.observation_space.contains(obs))
                if done or truncated:
                    self.assertIn('outcome',info)
                    self.assertGreaterEqual(info['decisions'],step+1)
                    return
            self.fail('Missing episode termination')
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()
