# Python client

This client uses **snakeoil3** set up an UDP connection on port 3101 to TORCS's _scr_server_.

Actions are sent to the _scr_server_, which in turn gives back the next state.

With the VTORCS modification the state can also be an image.

## Usage
The **TorcsEnv** class provides a simple abstraction of TORCS.

The API is similar to OpenAI gym, with some peculiarities:
```
from torcs_comp.torcs_comp import TorcsEnv

# filter and scale the returned state values
# dict with key: observation name, val: scale
state_filter = {}
state_filter["track"] = 200.0
state_filter["speedX"] = 300.0
state_filter["speedY"] = 300.0
state_filter["speedZ"] = 300.0
state_filter["wheelSpinVel"] = 1.0
state_filter["rpm"] = 10000

# Instantiate the environment
env = TorcsEnv(throttle = False, vision = True, state_filter = state_filter)
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

The key parts are:
```
env = TorcsEnv(throttle = False, vision = True, state_filter = state_filter)
```
* **_throttle = False_** sets automatic throttle control, for simpler training
* **_vision = True_** turns on RGB vision. This returns a 64x64x3 numpy array, always on the last position of the state list.
* **_state_filter = state_filter_** sets the selected sensors.

```
state_new, reward, terminal = env.step(action)
```
* **_state_new_** is the next state. A list made of all the selected "sensors" is returned.
* **_reward_** is the resulting reward fot the transition.
* **_terminal_** is set to true if the termination clause is verified.

## Further customization
The reward function and the termination clause can be customized by changing the content of the functions in **torcs_comp/_reward.py_** and **torcs_comp/_terminator.py_**.
