import unittest
import numpy as np
import torch
from training.engine import Game
from training.pv import PolicyValue,encode,augment,choose,fork,winners


class PolicyValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_both_sizes_and_separate_players(self):
        model=PolicyValue().eval()
        for size in (15,30):
            game=Game(3,size)
            for player in range(3):
                obs=encode(game,player)
                self.assertEqual(obs.shape,(14,size,size))
                self.assertEqual(obs[3:6].sum(),3)
                for channel in range(3):self.assertEqual(obs[channel].sum(),1)
                p,v,t=model(torch.from_numpy(obs[None]))
                self.assertEqual(p.shape,(1,4));self.assertEqual(v.shape,(1,3));self.assertEqual(t.shape,(1,3))
                self.assertTrue(torch.isfinite(v).all())

    def test_transforms_preserve_action_legality(self):
        game=Game(3,15);obs=encode(game,0)
        mask=game.mask(0);policy=mask/mask.sum();rng=np.random.default_rng(27)
        for _ in range(32):
            x,pi,m,z,t=augment(obs,policy,mask,np.array([1.,0,0]),np.array([.5,.2,.3]),rng)
            self.assertAlmostEqual(pi.sum(),1)
            self.assertEqual((pi[~m]).sum(),0)
            self.assertAlmostEqual(z.sum(),1);self.assertAlmostEqual(t.sum(),1)
            y,px=np.argwhere(x[3])[0]
            for d,(dx,dy) in enumerate(((0,-1),(1,0),(0,1),(-1,0))):
                if m[d]:self.assertTrue(x[0,y+dy,px+dx] or x[6,y+dy,px+dx])

    def test_search_private_and_legal(self):
        torch.manual_seed(7);game=Game(3,15);state=game.state()
        model=PolicyValue().eval()
        for depth in (0,1,3):
            move,pi=choose(model,game,1,np.random.default_rng(11),depth=depth,samples=2)
            self.assertTrue(game.mask(1)[move]);self.assertAlmostEqual(float(pi.sum()),1,places=6)
            self.assertEqual(game.state(),state)
        child=fork(game);child.blocks.add((1,1));child.board[1,1]=0
        self.assertEqual(game.state(),state)

    def test_tied_first_credit(self):
        game=Game(3,15);game.scores=[10,10,3]
        np.testing.assert_equal(winners(game),[.5,.5,0])

    def test_value_really_changes_search_decision(self):
        class DirectionValue(torch.nn.Module):
            def __init__(self,sign):super().__init__();self.sign=sign
            def forward(self,x):
                coordinate=(x[:,3]*x[:,11]).sum((1,2))*self.sign*5
                value=torch.stack([coordinate,coordinate*0,coordinate*0],1)
                return torch.zeros(len(x),4),value,torch.zeros(len(x),3)
        game=Game(3,15)
        right,_=choose(DirectionValue(1),game,0,np.random.default_rng(12),1,2)
        left,_=choose(DirectionValue(-1),game,0,np.random.default_rng(12),1,2)
        self.assertEqual(right,1);self.assertEqual(left,3)

    def test_release_screen_cannot_mislabel_partial_evaluation(self):
        from training.pv_run import cannot_reach_gate
        self.assertFalse(cannot_reach_gate(0,8,27))
        self.assertTrue(cannot_reach_gate(0,9,27))
        self.assertFalse(cannot_reach_gate(19,27,27))
        self.assertTrue(cannot_reach_gate(18,27,27))


if __name__=='__main__':unittest.main()
