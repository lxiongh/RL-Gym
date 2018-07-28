# load agent modules
from dqn_agent import DQNAgent

# load cv lib
import gym
import torch

from collections import deque
import numpy as np

from gym.wrappers import Monitor
import os
import sys
import getopt
import utils


def usage():
    """
    To training model

    Usage: run_train.py [-h|--help]

    """

def deque_append_list_elm(deque, list_data):
    for data in list_data:
        deque.append(data)

def main(argv=None):
    try:
        options, args = getopt.getopt(sys.argv[1:], "s:x:b:u:mh", [
                                      "step=", "max_eps=", "buffer_size=", "hidden_unit=","monitor", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        print(usage.__doc__)
        sys.exit(1)

    GAME_NAME = 'MountainCar-v0'
    AGENT_NAME = 'DQN-lr_1_e-3'
    MONITOR = False
    print_step = 10
    max_eps = 1000
    buffer_size=1000000
    hidden_unit = 64
    lr=3e-3

    print(options)
    for o, v in options:
        if o in ("-h", "--help"):
            print(usage.__doc__)
            sys.exit()
        elif o in ("-m", "--monitor"):
            MONITOR = True
        elif o in ("-s", "--step"):
            print_step = int(v)
        elif o in ("-x", "--max_eps"):
            max_eps = int(v)
        elif o in ("-b", "--buffer_size"):
            buffer_size = int(v)
        elif o in ("-u", "--hidden_unit"):
            hidden_unit = int(v)
        else:
            print(usage.__doc__)
            sys.exit()

    print('process game: %s\tusing agent: %s' % (GAME_NAME, AGENT_NAME))

    # -------------------------------- loop for training -----------------------------
    # preparing env
    output_dir = '%s/%s' % (GAME_NAME, AGENT_NAME)
    cmd = 'mkdir -p %s && mkdir -p %s/%s' % (GAME_NAME, GAME_NAME, AGENT_NAME)
    os.system(cmd)

    env = gym.make(GAME_NAME)
    if MONITOR:
        env = Monitor(env, directory=output_dir, force=True, video_callable=lambda ep: ep % 10 == 0, write_upon_reset=True, mode='training')
    
    env.seed(0)

    state_num = len(env.reset()) * 4
    print(state_num)
    action_sample = env.action_space.sample()
    action_num = env.action_space.n if isinstance(action_sample, int) else len(action_sample)
    print('state_num: %d\taction_num: %d' % (state_num, action_num))
    
    device = torch.device('cpu')
    agent = DQNAgent(state_num, action_num, buffer_size=buffer_size, batch_size=128, device=device, hidden_unit=hidden_unit, lr=lr)

    scores_window = deque(maxlen=print_step)  # last 10 scores
    avg_scores = []
    state_deque = deque(maxlen=8)

    for i_episode in range(max_eps):
        score = 0
        s = env.reset()
        deque_append_list_elm(state_deque, s)
        deque_append_list_elm(state_deque, s)
        deque_append_list_elm(state_deque, s)
        deque_append_list_elm(state_deque, s)
        iter = 0
        state = np.array(state_deque)
        while True:
            
            action = agent.choose_action(state)
            s_n, reward, done, _ = env.step(action)
            
            deque_append_list_elm(state_deque, s_n)
            next_state = np.array(state_deque)
            
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            iter += 1
            if done:
                break

        scores_window.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}\titer: {}'.format(
            i_episode, np.mean(scores_window), iter), end="")
        if i_episode % print_step == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\titer: {}'.format(
                i_episode, np.mean(scores_window), iter))
            # save model
            agent.save_model_params(output_dir, i_episode)

        avg_scores.append(np.mean(scores_window))
        sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    main()
