# Python client
* [Intro](#intro)
* [Usage](#usage)
    + [Config file](#config-file)
    + [Arguments](#arguments)
    + [Using a custom algorithm](#using-a-custom-algorithm)
    + [Using your run function](#using-your-run-function)
  * [Example](#example)
    + [Key parts](#key-parts)
  * [Further customization](#further-customization)

## Intro

This client uses **snakeoil3** set up an UDP connection on port 3001 to TORCS's _scr_server_.

Actions are sent to the _scr_server_, which in turn gives back the next state.

The python interface to torcs, **TorcsEnv**, can be used either with the rest of the launch file or as a standalone environment.

With this modified version of TORCS the state can also be an image.


## Usage
The **TorcsEnv** class by itself provides a simple abstraction of TORCS. The API is similar to OpenAI gym, with some peculiarities.

### Config file
The default config file is located at [simulation.yaml](config/simulation.yaml) and will look like this.
```yaml
# docker image name. set to 0 to run torcs on host
image_name : "gerkone/torcs"

# run module and path. change to your test script location and function
mod_name : "main"
run_path : "driver/run_torcs.py"

# run module and path. change here to your agent algorithm
algo_name: "DDPG"
algo_path: "driver/agents/ddpg/ddpg.py"

# agent hyperparameters
hyperparams:
  actor_lr : 0.000001
  critic_lr : 0.000003
  batch_size : 32
  gamma : 0.99
  rand_steps : 0
  buf_size : 10000
  tau : 0.001
  fcl1_size : 128
  fcl2_size : 64

# dict with key: observation name, val: normalization scale (sensor max value)
sensors:
  track : 200.0
  speedX : 300.0
  speedY : 300.0
  speedZ : 300.0
  wheelSpinVel : 1.0
  rpm : 10000
  # VISION SENSOR. set to turn vision on
  img: 1.0

# DO NOT CHANGE (for now)
img_width : 640
img_height : 480
```

### Arguments
The arguments are taken from the specified config file. A custom **_run/main_** function must account for these arguments:
- **verbose** if set prints some info from the simulation.
- **hyperparams** is a dictionary containing the hyperparameters for your agent.
- **sensors** is a dictionary describing the sensors and their scale.
- **image_name** sets the name of the docker image. Set to 0 to run TORCS on your host.
- **img_width**, **img_height** sets the size of the vision, if active.


### Using a custom algorithm
It is possible to use a custom algorithm. This is easily done by changing
```yaml
algo_name: "DDPG"
algo_path: "driver/agents/ddpg/ddpg.py"
```
- **algo_name** is the name of your algorithm class. The ___init___ method header should look like this:
```python
  def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):
    [...]
```
- **algo_path** is the relative path to the file containing your algorithm. Starting from the project root folder.

With these settings the function with name _main_ from the file _driver/run_torcs.py_ will be run when _python pytorcs.py_ is run.

After editing _simulation.yaml_ the environment can be started up as always with
```
python pytorcs.py
```

### Using your run function
The run function can also be easily changed. The two key parameters regarding the launch of your function are
```yaml
mod_name : "main"
run_path : "driver/run_torcs.py"
```
- **mod_name** is the name of your function
- **run_path** is the relative path to the file containing your run function. Starting from the project root folder.

With these settings the function with name _main_ from the file _driver/run_torcs.py_ will be run when _python pytorcs.py_ is run.

The important thing is that your custom run function needs to have its header like this
```python
def run(verbose, hyperparams, sensors, image_name, img_width, img_height):
  [...]
```

After editing _simulation.yaml_ the environment can be started up as always with
```
python pytorcs.py
```


## Example

```python
from torcs_client.torcs_comp import TorcsEnv
def run(verbose, hyperparams, sensors, image_name, img_width, img_height):
  env = TorcsEnv(throttle = False, verbose = verbose, state_filter = sensors)
  action_dims = [env.action_space.shape[0]]
  state_dims = [env.observation_space.shape[0]]  # sensors input
  action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

  agent = ...
  for i in range(N_EPISODES):
      # resets the environment to the initial state
      state = env.reset()
      terminal = False
      score = 0
      while not terminal:
          # predict new action
          action = agent.predict()

          # perform the transition according to the predicted action
          state_new, reward, terminal = env.step(action)

          # do stuff to the agent
          # e.g. save to replay buffer, agent.learn(i), ...

          # iterate to the next state
          state = new_state
```

A complete working example can be found in [run_torcs.py](https://github.com/gerkone/pyTORCS-docker/blob/master/driver/run_torcs.py)
### Key parts

1. 
```python
action_dims = [env.action_space.shape[0]]
state_dims = [env.observation_space.shape[0]]  # sensors input
action_boundaries = [env.action_space.low[0], env.action_space.high[0]]
 ```
 The **action and observation spaces** are OpenAI gym spaces that describe the action and state space respectively.

 Note that the observation space does not include the vision, only the other sensors.

2. 
```python
state_filter = sensors
```
The **sensors/state_filter** dictionary specifies which virtual sensor include in the state. The value is the normalization scale factor of the sensor.

3. 
```python
env = TorcsEnv(throttle = False, verbose = verbose, state_filter = state_filter)
```
  - **_throttle_** sets automatic throttle control, for simpler training. This means that the action space is restricted to just steering. The throttle is controlled via a simple cruise control.
  - **_state_filter__** sets the selected sensors.
  - **_verbose_** if set prints some info from the simulation.

4. 
```python
state_new, reward, terminal = env.step(action)
```
  - **_state_new_** is the next state. A list made of all the selected "sensors" is returned. The vision is in form of a 64x64x3 numpy array, always on the last position of the state list.
  - **_reward_** is the resulting reward fot the transition.
  - **_terminal_** is set to true if the termination clause is verified.

## Further customization
An example deep reinforcement learning agent, using DDPG, is provided.

The reward function and the termination clause can be customized by changing respectively **torcs_client/reward.py** and **torcs_client/terminator.py**.
