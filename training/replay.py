"""PPO followed by small teacher-replay updates to test imitation forgetting."""
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.imitate import augment


class ReplayPPO(MaskablePPO):
    replay_data=None
    _replay_cache=None

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['_replay_cache']

    def train(self):
        super().train()
        if not self.replay_data:
            return
        if self._replay_cache is None:
            with np.load(self.replay_data) as data:
                groups=data['episodes']; unique=np.unique(groups)
                held_out=np.random.default_rng(2026).choice(unique,max(1,len(unique)//10),replace=False)
                rng=np.random.default_rng(42)
                selected=[]
                for group in unique:
                    if group in held_out:
                        continue
                    ids=np.flatnonzero(groups==group)
                    selected.extend(rng.choice(ids,min(len(ids),256),replace=False))
                self._replay_cache=(data['observations'][selected],data['masks'][selected],
                                    data['actions'][selected])
        observations,masks,actions=self._replay_cache
        rng=np.random.default_rng((self.seed or 0)+self.num_timesteps)
        losses=[]
        for _ in range(8):
            ids=rng.integers(len(actions),size=128)
            obs=torch.as_tensor(observations[ids],dtype=torch.float32,device=self.device)
            mask=torch.as_tensor(masks[ids],device=self.device)
            labels=torch.as_tensor(actions[ids],device=self.device)
            obs,mask,labels=augment(obs,mask,labels,rng)
            distribution=self.policy.get_distribution(obs,action_masks=mask)
            loss=-distribution.log_prob(labels).mean()
            self.policy.optimizer.zero_grad()
            (.2*loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())
        self.logger.record('train/teacher_replay_nll',float(np.mean(losses)))
