from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from random_walk import RandomWalk
from dqn_agent_random_walk import DQNAgent_RandomWalk

def init():
    img.set_array(state_t_1)
    plt.axis("off")
    return img,

def animate(step):
    global state_t_1, reward_t

    state_t = state_t_1

    # execute action in environment
    action_t = agent.select_action(state_t, 0.0)
    env.execute_action(action_t)

    # observe environment
    state_t_1, reward_t = env.observe()

    # animate
    img.set_array(state_t_1)
    plt.axis("off")
    return img,

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    # environmet, agent
    env = RandomWalk()
    agent = DQNAgent_RandomWalk(env.enable_actions, env.name)
    agent.load_model(args.model_path)

    # variables
    state_t_1, reward_t = env.observe()

    # animate
    fig = plt.figure(figsize=(env.screen_n_rows / 2, env.screen_n_cols / 2))
    fig.canvas.set_window_title("{}-{}".format(env.name, agent.name))
    img = plt.imshow(state_t_1, interpolation="none", cmap="gray")
    ani = animation.FuncAnimation(fig, animate, init_func=init, blit=True)

    if args.save:
        # save animation (requires ImageMagick)
        ani_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tmp", "demo-{}.gif".format(env.name))
        ani.save(ani_path, writer="imagemagick")
    else:
        # show animation
        plt.show()