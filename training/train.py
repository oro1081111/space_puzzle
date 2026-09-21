import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

# Bound native math threads before imports, including spawned workers.
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from training.engine import RULE_VERSION
from training.env import TrainingEnv, ExpertTrainingEnv
from training.model import BoardCNN
from training.verify import ROOT


class Progress(BaseCallback):
    def __init__(self, directory, interval=16384):
        super().__init__()
        self.directory=directory
        self.interval=interval
        self.next_save=interval
        self.episodes=0
        self.started=time.perf_counter()
        self.last_report=0
        self.pause_requested=False

    def _on_step(self):
        if (self.directory/'PAUSE').exists():
            self.pause_requested=True
            print('Pause requested; saving latest checkpoint and closing workers.',flush=True)
            return False
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            if done:
                self.episodes+=1
                with (self.directory/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({k:v for k,v in info.items() if k not in ('terminal_observation',)},default=float)+'\n')
        if self.num_timesteps>=self.next_save:
            name=self.directory/f'snapshot_{self.num_timesteps:09d}.zip'
            temp=self.directory/'saving.zip'
            self.model.save(temp)
            temp.replace(name)
            self.next_save=self.num_timesteps+self.interval
        if time.perf_counter()-self.last_report>=20:
            report=dict(steps=self.num_timesteps,episodes=self.episodes,
                        elapsed_seconds=round(time.perf_counter()-self.started,1))
            (self.directory/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report),flush=True)
            self.last_report=time.perf_counter()
        return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=100352)
    parser.add_argument('--envs',type=int,default=2)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--size',type=int,choices=[15,30],default=15)
    parser.add_argument('--run',default='training/runs/pilot15')
    parser.add_argument('--resume')
    parser.add_argument('--subprocess',action='store_true')
    parser.add_argument('--expert',action='store_true')
    parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--replay-data')
    args=parser.parse_args()
    if args.replay_data and not args.resume:
        parser.error('--replay-data requires --resume from a pretrained policy')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    priority='normal'
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        if kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000):
            priority='below_normal'
    directory=Path(args.run).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'PAUSE').exists():
        raise RuntimeError('This run is paused. Remove its PAUSE file only when resuming is authorized.')
    verification_name='verification.json' if args.size==30 else f'verification{args.size}.json'
    verification=json.loads((ROOT/'training/runs'/verification_name).read_text())
    digest=hashlib.sha256((ROOT/'grid-clash/index.html').read_bytes()).hexdigest()
    if not verification['passed'] or verification['source_sha256']!=digest:
        raise RuntimeError('Run python -m training.verify for the current browser source first')
    if args.replay_data:
        provenance=json.loads(Path(args.replay_data).with_suffix('.json').read_text())
        if provenance['source_sha256']!=digest:
            raise RuntimeError('Replay dataset uses different browser rules')
    def make_env():
        torch.set_num_threads(1 if args.subprocess else args.threads)
        env_class=ExpertTrainingEnv if args.expert else TrainingEnv
        return Monitor(env_class(pool=directory,shaping=.1,size=args.size))
    vec=(SubprocVecEnv if args.subprocess else DummyVecEnv)([make_env for _ in range(args.envs)])
    config=dict(vars(args),rule_version=RULE_VERSION,source_sha256=digest,priority=priority,
                player_scope=[2,3,4],current_players=2,
                algorithm='PPO+teacher replay' if args.replay_data else 'MaskablePPO',gamma=.999,gae_lambda=.98,shaping=.1,
                packages={p:importlib.metadata.version(p) for p in
                          ['torch','numpy','scipy','gymnasium','pettingzoo','stable-baselines3','sb3-contrib']})
    config_file=directory/'config.json'
    if config_file.exists():
        with (directory/'config-history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(json.loads(config_file.read_text(encoding='utf-8')))+'\n')
    config_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
    if args.resume:
        override=dict(learning_rate=args.learning_rate) if args.learning_rate else {}
        if args.replay_data:
            from training.replay import ReplayPPO
            model=ReplayPPO.load(args.resume,env=vec,device='cpu',**override)
            model.replay_data=str(Path(args.replay_data).resolve())
        else:
            model=MaskablePPO.load(args.resume,env=vec,device='cpu',**override)
    else:
        model=MaskablePPO('CnnPolicy',vec,device='cpu',seed=args.seed,
            n_steps=256,batch_size=256,n_epochs=4,learning_rate=3e-4,
            gamma=.999,gae_lambda=.98,ent_coef=.02,target_kl=.03,
            policy_kwargs=dict(features_extractor_class=BoardCNN,normalize_images=False,
                               net_arch=dict(pi=[64],vf=[64])),verbose=0)
        model.save(directory/'untrained.zip')
    start=time.perf_counter()
    initial_steps=model.num_timesteps
    callback=Progress(directory)
    callback.next_save=model.num_timesteps+callback.interval
    try:
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(str(directory/'metrics'),['csv']))
        model.learn(total_timesteps=args.steps,callback=callback,reset_num_timesteps=not bool(args.resume))
    finally:
        model.save(directory/'latest.zip')
        vec.close()
    report=dict(completed=not callback.pause_requested,paused=callback.pause_requested,
                steps=model.num_timesteps,added_steps=model.num_timesteps-initial_steps,seconds=time.perf_counter()-start,
                episodes=callback.episodes,parameters=sum(p.numel() for p in model.policy.parameters()))
    (directory/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
