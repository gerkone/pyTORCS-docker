from torcs_client.torcs_comp import TorcsEnv

def main(verbose = False, hyperparams = None, sensors = None, image_name = "gerkone/torcs", driver = None,
        privileged = False, training = None, algo_name = None, algo_path = None, stack_depth = 1, img_width = 640, img_height = 480):

    max_steps = 1000
    n_epochs = 5
    episodes = 1000
    train_req = 1000

    if "max_steps" in training.keys(): max_steps = training["max_steps"]
    if "epochs" in training.keys(): n_epochs = training["epochs"]
    if "episodes" in training.keys(): episodes = training["episodes"]
    if "train_req" in training.keys(): train_req = training["train_req"]

    track_list = [None]
    car = None

    if "track" in training.keys(): track_list = training["track"]
    if "car" in training.keys(): car = training["car"]

    if driver != None:
        sid = driver["sid"]
        ports = driver["ports"]
        driver_id = driver["index"]
        driver_module = driver["module"]
    else:
        sid = "SCR"
        ports = [3001]
        driver_id = "0"
        driver_module = "scr_server"

    # Instantiate the environment
    env = TorcsEnv(throttle = training["throttle"], gear_change = training["gear_change"], car = car,
            verbose = verbose, state_filter = sensors, target_speed = training["target_speed"], sid = sid,
            ports = ports, driver_id = driver_id, driver_module = driver_module, image_name = image_name,
            privileged = privileged, img_width = img_width, img_height = img_height)

    test_env = TorcsEnv(throttle = training["throttle"], gear_change = training["gear_change"], car = car,
            verbose = verbose, state_filter = sensors, target_speed = training["target_speed"], sid = sid,
            ports = ports, driver_id = driver_id, driver_module = driver_module, image_name = image_name,
            privileged = privileged, img_width = img_width, img_height = img_height)

    action_dims = [env.action_space.shape[0]]
    state_dims = [env.observation_space.shape[0]]  # sensors input
    action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

    args = {}

    args["test_interval"] = 20480
    args["save_summary_interval"] = 20480
    args["save_model_interval"] = 20480
    args["max_steps"] = training["max_steps"]
    args["episode_max_steps"] = int(1e7)
    args["dir_suffix"] = ""
    if "model_dir" in hyperparams.keys():
        args["model_dir"] = hyperparams["model_dir"]
    args["test_interval"] = hyperparams["test_interval"]
    args["test_episodes"] = 1000

    if "track" in training.keys(): track_list = training["track"]

    if training["algo"] == "PPO":
        from agents.tf2rl.algos.ppo import PPO
        from agents.tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
        from agents.tf2rl.experiments.utils import load_expert_traj

        agent = PPO(
            state_shape = env.observation_space.shape,
            action_dim = env.action_space.high.size,
            is_discrete = False,
            max_action = env.action_space.high[0],
            batch_size = hyperparams["batch_size"],
            actor_units = (hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            critic_units = (hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            n_epoch = n_epochs,
            lr_actor = hyperparams["actor_lr"],
            lr_critic = hyperparams["critic_lr"],
            hidden_activation_actor = "tanh",
            hidden_activation_critic = "tanh",
            discount = hyperparams["gamma"],
            lam = hyperparams["lam"],
            vfunc_coef = hyperparams["c_1"],
            entropy_coef = hyperparams["c_2"],
            horizon = hyperparams["horizon"]
        )

        expert_trajs = None

        if "dataset_dir" in hyperparams.keys():
            expert_trajs = load_expert_traj(hyperparams["dataset_dir"])

        trainer = OnPolicyTrainer(agent, env, args, test_env = test_env, expert_trajs = expert_trajs)

    elif training["algo"] == "DDPG":
        from agents.tf2rl.algos.ddpg import DDPG
        from agents.tf2rl.experiments.trainer import Trainer

        agent = DDPG(
            state_shape = env.observation_space.shape,
            action_dim = env.action_space.high.size,
            memory_capacity = hyperparams["buf_size"],
            max_action = env.action_space.high[0],
            batch_size = hyperparams["batch_size"],
            actor_units = (hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            critic_units = (hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            lr_actor = hyperparams["actor_lr"],
            tau = hyperparams["tau"],
            lr_critic = hyperparams["critic_lr"],
            n_warmup = hyperparams["n_warmup"],
            update_interval = hyperparams["update_interval"]
        )

        trainer = Trainer(agent, env, args, test_env=test_env)
    elif training["algo"] == "GAIL":
        from agents.tf2rl.algos.ddpg import DDPG
        from agents.tf2rl.algos.gail import GAIL
        from agents.tf2rl.experiments.irl_trainer import IRLTrainer
        from agents.tf2rl.experiments.utils import load_expert_traj

        agent = DDPG(
            state_shape = env.observation_space.shape,
            action_dim = env.action_space.high.size,
            memory_capacity = hyperparams["buf_size"],
            max_action = env.action_space.high[0],
            batch_size = hyperparams["batch_size"],
            actor_units = (hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            critic_units = (hyperparams["fcl1_size"], hyperparams["fcl2_size"]),
            lr_actor = hyperparams["actor_lr"],
            tau = hyperparams["tau"],
            lr_critic = hyperparams["critic_lr"],
            n_warmup = hyperparams["n_warmup"],
            update_interval = hyperparams["update_interval"]
        )

        discriminator = GAIL(
            state_shape = env.observation_space.shape,
            action_dim = env.action_space.high.size,
            units = [100, 100],
            batch_size = hyperparams["batch_size"]
        )
        expert_trajs = load_expert_traj(hyperparams["dataset_dir"])
        trainer = IRLTrainer(agent, env, args, discriminator, expert_trajs["state"], expert_trajs["state_new"], expert_trajs["action"], test_env)

    trainer(track_list)

    input("All done")
