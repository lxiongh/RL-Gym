import gym
from gym.wrappers import Monitor

from dqn_agent import DQNAgent

env = gym.make('LunarLander-v2')
env = Monitor(env, directory='./play', force=True, video_callable=lambda ep: True, write_upon_reset=True, mode='training')
env.seed(0)
env.reset()

state_num = len(env.reset())
action_sample = env.action_space.sample()
action_num = env.action_space.n if isinstance(action_sample, int) else len(action_sample)

agent = DQNAgent(state_num, action_num, buffer_size=1000000, batch_size=128, hidden_unit=16, lr=1e-3)
agent.load_model_params('LunarLander-v2/DQN-lr_1_e-3',1740)

for i_eps in range(10):
    state = env.reset()
    while True:
        action = agent.choose_opt_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done==True:
            break
        