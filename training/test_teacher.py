import unittest
import itertools
import json
from training.verify import Oracle, ROOT
from training.teacher_eval import summarize
from training.engine import Game


class TeacherTests(unittest.TestCase):
    def test_generated_route_worker_matches_training_games(self):
        weights=json.loads((ROOT/'grid-clash/atlas-route-v2.json').read_text())['weights']
        left,right=Oracle(),Oracle()
        try:
            for seat in range(3):
                styles=['apex','sweep'];styles.insert(seat,'candidate')
                left.call(op='reset',size=15,count=3,seed=280001,styles=styles,weights=weights)
                styles[seat]='atlas'
                right.call(op='reset',size=15,count=3,seed=280001,styles=styles)
                for _ in range(2700):
                    expected=left.call(op='play',actions=['ai']*3)
                    actual=right.call(op='route',players=[seat])
                    self.assertEqual(expected,actual)
                    if actual['ended']:break
                self.assertTrue(actual['ended'])
        finally:left.close();right.close()

    def test_strength_gate_needs_full_matrix_and_every_group(self):
        rows=[dict(seed=seed,pair=pair,seat=seat,win=True,tie=False,finished=True)
              for seed,pair,seat in itertools.product(range(20),itertools.product(['apex','bastion','sweep'],repeat=2),range(3))]
        result=summarize(rows,540)
        self.assertTrue(result['strength_gate_passed'])
        self.assertFalse(result['release_approved'])
        self.assertFalse(summarize(rows[:27],540)['strength_gate_passed'])
        self.assertFalse(summarize([rows[0]]*540,540)['strength_gate_passed'])
        for r in rows:
            if r['pair']==('sweep','sweep'):r['win']=False
        self.assertGreater(summarize(rows,540)['win_rate'],.7)
        self.assertFalse(summarize(rows,540)['strength_gate_passed'])

    def test_candidate_complete_games_rule_parity(self):
        oracle=Oracle()
        try:
            for size,cuts in ((15,1),(30,1),(30,0)):
                game=Game(count=3,size=size)
                game.load(oracle.call(op='reset',size=size,count=3,seed=5101,
                    styles=['candidate','bastion','sweep'],weights=dict(cuts=cuts,topology=1,raceCost=5,avoidClash=1,scaledPhase=1,beamRace=2)))
                attempts=0
                while not game.ended and attempts<12*size*size:
                    result=oracle.call(op='play',actions=['ai']*3)
                    self.assertEqual(game.step(result['actions']),result['advanced'])
                    for key,value in game.state().items():self.assertEqual(value,result[key],key)
                    attempts+=1
                self.assertTrue(game.ended)
        finally:oracle.close()

    def test_probe_restores_state_memory_and_rng(self):
        left, right = Oracle(), Oracle()
        try:
            for size in (15, 30):
                opts = dict(op='reset', size=size, count=3, seed=192, styles=['apex', 'bastion', 'sweep'])
                state = left.call(**opts)
                self.assertEqual(state, right.call(**opts))
                for _ in range(3):
                    probe = left.call(op='teacher', player=0, probe=True, options=dict(horizon=4))
                    self.assertEqual({k: probe[k] for k in state}, state)
                    a = left.call(op='play', actions=['ai']*3)
                    b = right.call(op='play', actions=['ai']*3)
                    self.assertEqual(a, b)
                    state = {k: a[k] for k in state}
        finally:
            left.close();right.close()

    def test_no_opponent_strategy_peeking(self):
        oracle = Oracle()
        try:
            results = []
            for styles in (['apex']*3, ['apex', 'sweep', 'bastion']):
                oracle.call(op='reset', size=15, count=3, seed=71, styles=styles)
                results.append(oracle.call(op='teacher', player=0, probe=True, options=dict(horizon=8))['choice'])
            self.assertEqual(results[0], results[1])
        finally:
            oracle.close()

    def test_planning_rng_independent_of_live_rng(self):
        oracle = Oracle()
        try:
            choices = []
            for live_seed in (1, 1234567):
                oracle.call(op='reset', size=15, count=3, seed=live_seed,
                            styles=['candidate', 'apex', 'sweep'],
                            weights=dict(cuts=1, topology=1, raceCost=5))
                choices.append(oracle.call(op='teacher', player=0, probe=True,
                    options=dict(candidate=True, topCuts=2, horizon=8, profiles=3, seed=101))['choice'])
            self.assertEqual(choices[0], choices[1])
        finally:
            oracle.close()

    def test_pruning_preserves_selected_route(self):
        oracle=Oracle()
        try:
            oracle.call(op='reset',size=15,count=3,seed=25,styles=['candidate','apex','sweep'],
                        weights=dict(cuts=1,topology=1,raceCost=5))
            for _ in range(3):
                options=dict(candidate=True,mixedPlans=True,topCuts=2,horizon=12,profiles=9,seed=101)
                full=oracle.call(op='teacher',player=0,probe=True,options=options)['choice']
                fast=oracle.call(op='teacher',player=0,probe=True,options=dict(options,prune=True))['choice']
                self.assertEqual(full['action'],fast['action'])
                if full['rows']:
                    self.assertEqual(max(full['rows'],key=lambda r:r['score'])['plan'],
                                     max(fast['rows'],key=lambda r:r['score'])['plan'])
                oracle.call(op='play',actions=['ai']*3)
        finally:oracle.close()

    def test_teacher_replans_after_fixed_turn_callback(self):
        oracle=Oracle()
        try:
            oracle.call(op='reset',size=15,count=3,seed=910001,
                styles=['apex','candidate','apex'],weights=dict(cuts=1,topology=1,raceCost=5))
            for _ in range(4):
                result=oracle.call(op='teacher',player=1,options=dict(candidate=True,topCuts=1,horizon=8,profiles=3))
                self.assertIsNotNone(result['actions'][1], 'Previous fixed callback must not suppress replanning')
        finally:oracle.close()

    def test_teacher_matches_real_fallback_when_risk_blocks_frontier(self):
        oracle=Oracle()
        try:
            board=[[2]*15 for _ in range(15)]
            board[0][0]=board[1][0]=board[1][1]=0
            board[1][2]=-1;board[1][3]=1
            oracle.call(op='reset',size=15,count=3,seed=19,board=board,
                positions=[[0,0],[3,1],[14,14]],styles=['candidate','apex','apex'],weights=dict(cuts=1,avoidClash=1))
            result=oracle.call(op='teacher',player=0,probe=True,options=dict(candidate=True,topCuts=1,horizon=8,profiles=3))
            self.assertEqual(result['choice']['action'],2)
        finally:oracle.close()

    def test_opening_book_only_applies_to_fresh_candidate(self):
        oracle,control=Oracle(),Oracle()
        try:
            options=dict(size=15,count=3,seed=42,styles=['candidate','apex','sweep'],
                         weights=dict(openings={'0':dict(plan=[1,1],style='candidate')}))
            state=oracle.call(op='reset',**options)
            moved=oracle.call(op='play',actions=['ai']*3)
            self.assertEqual(moved['actions'][0],1)
            # Loading an arbitrary midgame board must not inject an opening.
            options.update(board=state['board'],positions=state['positions'])
            restored=oracle.call(op='reset',**options)
            self.assertEqual(restored['board'],state['board'])
            control.call(op='reset',**dict(options,weights={}))
            game=Game(count=3,size=15);game.load(restored)
            actual=oracle.call(op='play',actions=['ai']*3)
            self.assertEqual(actual,control.call(op='play',actions=['ai']*3))
            game.step(actual['actions'])
            self.assertEqual(game.state()['board'],actual['board'])
        finally:oracle.close();control.close()


if __name__ == '__main__':
    unittest.main()
