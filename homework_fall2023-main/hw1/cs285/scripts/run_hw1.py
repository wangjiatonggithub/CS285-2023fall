"""
Runs behavior cloning and DAgger for homework 1

Functions to edit:
    1. run_training_loop
"""


import pickle #序列化库
import os
import time
import gym

import numpy as np
import torch

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]


def run_training_loop(params):
    """
    Runs training with the specified parameters
    (behavior cloning or dagger)

    Args:
        params: experiment parameters
    """

    #############
    ## INIT
    #############

    # Get params, create logger, create TF session
    logger = Logger(params['logdir'])

    # Set random seeds
    seed = params['seed'] # 设置随机数种子以便环境可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    # Set logger attributes
    log_video = True
    log_metrics = True

    #############
    ## ENV
    #############

    # Make the gym environment
    env = gym.make(params['env_name'], render_mode=None)
    env.reset(seed=seed)

    # Maximum length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps # episode最大长度，从命令行获取或用环境自带的值
    MAX_VIDEO_LEN = params['ep_len'] # 视频最大帧数，与episode最大长度相等

    assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if 'model' in dir(env): # 获取视频保存频率
        fps = 1/env.model.opt.timestep
    else:
        fps = env.env.metadata['render_fps']

    #############
    ## AGENT
    #############

    # TODO: Implement missing functions in this class.
    actor = MLPPolicySL( # 调用MLP_policy函数
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['size'],
        learning_rate=params['learning_rate'],
    )

    # replay buffer
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size']) # 调用relpay_buffer函数

    #######################
    ## LOAD EXPERT POLICY
    #######################

    print('Loading expert policy from...', params['expert_policy_file']) # 加载专家策略，并移动到当前设备
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file']) # 调用loaded_gaussian_policy函数
    expert_policy.to(ptu.device) # 调用pytorch_util函数
    print('Done restoring expert policy...')

    #######################
    ## TRAINING LOOP
    #######################

    # init vars at beginning of training
    total_envsteps = 0 # DAgger算法中与环境交互的总步数
    start_time = time.time()

    # 每次迭代都是先采集一次数据在训练n次
    for itr in range(params['n_iter']):
        print("\n\n********** Iteration %i ************"%itr)

        # decide if videos should be rendered/logged at this iteration
        log_video = ((itr % params['video_log_freq'] == 0) and (params['video_log_freq'] != -1))
        # decide if metrics should be logged
        log_metrics = (itr % params['scalar_log_freq'] == 0)

        print("\nCollecting data to be used for training...")
        if itr == 0: # 第1次迭代，从原始专家数据集中采集，BC算法只有1次迭代，DAgger算法有多次迭代
            # BC training from expert data.
            paths = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0
        else: # 之后每次迭代DAgger算法都要重标注采集数据并放入经验回放池
            # DAGGER training from sampled data relabeled by expert
            assert params['do_dagger']
            # TODO: collect `params['batch_size']` transitions
            # HINT: use utils.sample_trajectories
            # TODO: implement missing parts of utils.sample_trajectory
            # paths, envsteps_this_batch = TODO
            paths, envsteps_this_batch = utils.sample_trajectories(env, actor, params['batch_size'], params['ep_len'])

            # relabel the collected obs with actions from a provided expert policy
            if params['do_dagger']:
                print("\nRelabelling collected observations with labels from an expert policy...")

                # TODO: relabel collected obsevations (from our policy) with labels from expert policy
                # HINT: query the policy (using the get_action function) with paths[i]["observation"]
                # and replace paths[i]["action"] with these expert labels
                # paths = TODO
                for path in paths:
                    obs = path["observation"]
                    # 支持专家策略返回单个 action 或 (action, info)
                    actions = []
                    for ob in obs:
                        a = expert_policy.get_action(ob)
                        if isinstance(a, tuple):
                            a = a[0]
                        actions.append(a)
                    path["action"] = np.array(actions)

        total_envsteps += envsteps_this_batch
        # add collected data to replay buffer
        replay_buffer.add_rollouts(paths)

        # train agent (using sampled data from replay buffer)
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []
        for _ in range(params['num_agent_train_steps_per_iter']):

          # TODO: sample some data from replay_buffer
          # HINT1: how much data = params['train_batch_size']
          # HINT2: use np.random.permutation to sample random indices
          # HINT3: return corresponding data points from each array (i.e., not different indices from each array)
          # for imitation learning, we only need observations and actions.  
          # ob_batch, ac_batch = TODO
          num_data_points = replay_buffer.obs.shape[0]
          indices = np.random.choice(num_data_points, size=params['train_batch_size'], replace=True)
          ob_batch = replay_buffer.obs[indices]
          ac_batch = replay_buffer.acs[indices]

          # use the sampled data to train an agent
          train_log = actor.update(ob_batch, ac_batch)
          training_logs.append(train_log)

        # log/save
        print('\nBeginning logging procedure...')
        if log_video:
            # save eval rollouts as videos in tensorboard event file
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories( # 采集评估视频
                env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            # save videos
            if eval_video_paths is not None:
                logger.log_paths_as_videos(
                    eval_video_paths, itr,
                    fps=fps,
                    max_videos_to_save=MAX_NVIDEO,
                    video_title='eval_rollouts')

        if log_metrics:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories( # 采集评估轨迹
                env, actor, params['eval_batch_size'], params['ep_len'])

            logs = utils.compute_metrics(paths, eval_paths) # 计算评估指标
            # compute additional metrics
            logs.update(training_logs[-1]) # Only use the last log for now
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            logger.flush()

        if params['save_params']:
            print('\nSaving agent params')
            actor.save('{}/policy_itr_{}.pt'.format(params['logdir'], itr))


def main():
    import argparse #参数解析器
    parser = argparse.ArgumentParser() # required指定该参数是否必须提供，action指定解析到的参数如何处理
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int) # episode最长步数，若命令行没给，就用环境自带的值

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5) # 存储评估视频的频率
    parser.add_argument('--scalar_log_freq', type=int, default=1) # 存储评估标量指标的频率
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args) # 将上述参数设置存入params字典

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    # directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir # 存储日志的路径
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    run_training_loop(params)


if __name__ == "__main__":
    main()
