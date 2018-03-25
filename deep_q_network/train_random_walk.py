#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from random_walk import RandomWalk
from dqn_agent_random_walk import DQNAgent_RandomWalk

if __name__ == "__main__":
    print"start"

    # parameters
    n_epochs = 100
    Q_max = 0.0

    # environment:RandomWalk/agent:DQNAgent_RandomWalk
    env = RandomWalk()
    agent = DQNAgent_RandomWalk(env.enable_actions, env.name)

    state_t_1, reward_t = env.observe()

    for e in range(n_epochs):

        state_t = state_t_1

        # execute action in environment
        action_t = agent.select_action(state_t, agent.exploration)
        env.execute_action(action_t)

        # observe environment
        state_t_1, reward_t = env.observe()

        # store experience
        agent.store_experience(state_t, action_t, reward_t, state_t_1)

        # experience replay
        agent.experience_replay()

        # for log
        Q_max = np.max(agent.Q_values(state_t))
        env.Q_save(Q_max)

        print("EPOCH: {:03d}/{:03d} | Q: {:.4f} ".format(e, n_epochs-1, Q_max ))

        # save model
        if e%100 == 0:
            agent.save_model()

    # save model
    agent.save_model()