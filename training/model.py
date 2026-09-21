import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BoardCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space,features_dim)
        spatial=(observation_space.shape[-1]+3)//4
        self.net=nn.Sequential(
            nn.Conv2d(7,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Flatten(),nn.Linear(32*spatial*spatial,features_dim),nn.ReLU())

    def forward(self, observations):
        return self.net(observations)


class HeadAwareCNN(BaseFeaturesExtractor):
    """Preserve exact head-local features alongside coarse whole-board context."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space,features_dim)
        self.conv=nn.Sequential(nn.Conv2d(7,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.pool=nn.AdaptiveAvgPool2d((5,5))
        self.out=nn.Sequential(nn.Linear(32*25+32,features_dim),nn.ReLU())

    def forward(self, observations):
        features=self.conv(observations)
        local=(features*observations[:,3:4]).sum(dim=(2,3))
        return self.out(torch.cat([local,self.pool(features).flatten(1)],dim=1))
