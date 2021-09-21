#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger



def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, bc=False):
    exp_data = np.load('/home/yzc/Desktop/safe-rl/BC+CPO/scripts/expert_data/expert_data_pointgoal1_cpo.npz')
    exp_state = exp_data['s']
    exp_action = exp_data['a']
    threshold = 0.5

    def gaussian_likelihood(x, mu, log_std):
        # std = np.exp(log_std)
        # p = 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))
        pre_sum = -0.5 * (((x - mu) / (np.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return np.sum(pre_sum, axis=1)

    def compute_dis_cost(obs_list, act_list):
        action_list = []
        for s in obs_list:
            diff = s - exp_state
            dis = np.sum(diff ** 2, axis=1)
            closest_index = np.argmin(dis)
            action_list.append(exp_action[closest_index])
        exp_act = np.array(action_list)

        # gauss=N(mu,log_std)
        # mu = np.array([info['mu'][0] for info in pi_info_list])
        # log_std = np.array([info['log_std'] for info in pi_info_list])
        # log_p = gaussian_likelihood(exp_act, mu, log_std)
        # p = np.exp(log_p)

        # gauss=N(exp_act,std=0.1)
        log_std = np.ones_like(exp_act) * np.log(0.5)
        log_p = gaussian_likelihood(np.array(act_list), exp_act, log_std)
        p = np.exp(log_p)

        cost = np.array(p <= threshold, dtype=np.float32)

        return cost

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("
    if bc:
        obs, acts, rewards, costs = [], [], [], []

    obs_list = []
    act_list = []

    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o_, r, d, info = env.step(a)

        obs_list.append(o)
        act_list.append(a)

        # collect data
        if bc:
            obs.append(o)
            acts.append(a)
            rewards.append(r)
            costs.append(info['cost'])

        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        o = o_
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1
            cost_list = compute_dis_cost(obs_list, act_list)
            print('EpActCost:', np.sum(cost_list))

    if bc:
        import os
        if not os.path.exists('expert_data'):
            os.makedirs('expert_data')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, 'expert_data/expert_data_pointgoal1_cpo')
        np.savez(save_path, s=np.array(obs), a=np.array(acts), r=np.array(rewards), c=np.array(costs))


    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='/home/yzc/Desktop/safe-rl/BC+CPO/expert/2021-09-10_cpo_PointGoal1/2021-09-10_18-16-44-cpo_PointGoal1_s0')
    parser.add_argument('--len', '-l', type=int, default=1000)
    parser.add_argument('--bc', type=bool, default=False)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', type=bool, default=True)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', type=bool, default=True)
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender), args.bc)
