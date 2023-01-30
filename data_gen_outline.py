import pandas as pd
import numpy as np
import random
import toml

agents = 3
items = 100
agent_dist = [0, .25, .25, .5]
users = 40
compatibility1 = "0,0,1"
compatibility2 = "1,0,0"
compatibility3 = "1,0,1"
dist1 = [0.33,.33,.33]
start1 = 1
end = 20
dist2 = [.8,.2,0]
start2 = 21

class item_user_gen:
    def __init__(self, config_file):
        self.config = toml.load(config_file)
        self.agents = self.config["agents"]
        self.items = self.config["items"]
        self.agent_dist = self.config["agent_dist"]
        self.users = self.config["users"]
        self.compatibility1 = self.config["compatibility1"]
        self.compatibility2 = self.config["compatibility2"]
        self.compatibility3 = self.config["compatibility3"]


def generate_users(self):
        users = pd.DataFrame(columns=['id', 'compatabilities'])
        compatabilities = [self.compatibility1, self.compatibility2, self.compatibility3]
        for i in range(1, self.users+1):
            compatability = random.choice(compatabilities)
            users = users.append({'id': i, 'compatabilities': compatability}, ignore_index=True)
        return users

def generate_items(self):
        items = pd.DataFrame(columns = ['id', 'agents'])
        for i in range(1, self.item+1):
            agent = random.choices(range(0,1+self.agents), self.agent_dist)
            items = items.append({'id': i, 'agents': agent}, ignore_index=True)
        return items

class regime_gen:
    def __init__(self, config_file):
        self.config = toml.load(config_file)   
        self.dist1 = self.config["dist1"]
        self.start1 = self.config["start1"]
        self.end = self.config["end"]
        self.dist2 = self.config["dist2"]
        self.start2 = self.config["start2"]
        
def generate_regime(self):
        regime = pd.DataFrame()
        compatabilities = [self.compatibility1, self.compatibility2, self.compatibility3]
        for i in range(self.start1, self.end+1):
            user = random.choices(compatabilities, self.dist1)
            regime = regime.append(user, ignore_index=True)
        for i in range(self.start2, self.users+1):
            user = random.choices(compatabilities, self.dist2)
            regime = regime.append(user, ignore_index=True)
        return regime
