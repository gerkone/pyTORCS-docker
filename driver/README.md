# Driver

The **_driver_** is made up of two components:
* **torcs_client** is the compatibility client that allows for TORCS abstraction.
* **agents** is the folder designated to contain the autonomous agents

## Action space
The action space is variable in size, depending on how the environment is configured.

- If _throttle == False_
  The action space is a 1-dimensional array, ranging from [-1,1]. The single value represents the steering request (-1 complete left, 1 complete right).
  ```
  action[0] := steering
  ```

- If _throttle == True_
  The action space is a 2-dimensional array. The first value remains the steering.

  The second value is combined brake/throttle request:
  + [-1,0] for the brake
  + [0,1] for the throttle.
  ```
  action[0] := steering
  action[1] := breaking/throttle
  ```

## Sensor description
- **_track_** - Vector of 19 range finder sensors: each sensors returns the distance between the track edge and the car within a range of 200 meters.
- **_angle_** - Angle between the car direction and the direction of the track axis.
- **_speedX_** - Speed of the car along its longitudinal axis.
- **_speedY_** - Speed of the car along its transverse axis.
- **_speedZ_** - Speed of the car along its Z axis.
- **_wheelSpinVel_** - Vector of 4 sensors representing the rotation speed of wheels.
- **_rpm_** - Number of rotation per minute of the car engine (could be used by the gearshift).
- **_trackPos_** - Distance between the car and the track axis.
- **_img_** - Vision from the driver perspective. By default a rgb 640x480 image.

Many other values are sent by torcs, but most of them are (currently) not useful for our purpose. The full list is:
```
angle
curLapTim
damage
distFromStart
distRaced
fuel
gear
lastLapTime
opponents
racePos
rpm
speedX
speedY
speedZ
track
trackPos
wheelSpinVel
z
```

More info on the sensors can be found in the original [scr paper](https://arxiv.org/pdf/1304.1672.pdf).

## State space
The state space is variable too. Its size and composition can be changed in the [simulation.yaml](config/simulation.yaml) configuration file.

The returned observation at each step is a dictionary made up of numpy arrays of (normalized) sensor values, with the sensor name as key.

For example an observation may look like this:
```
|observation["track"]| = [19]
|observation["angle"]| = [1]
|observation["speedX"]| = [1]
|observation["speedY"]| = [1]
|observation["speedZ"]| = [1]
|observation["wheelSpinVel"]| = [4]
|observation["rpm"]| = [1]
|observation["trackPos"]| = [1]
|observation["img"]| = [640, 480, 3]
```

## Customization
To use the environment with your custom run file or use it standalone follow [this guide](https://github.com/gerkone/pyTORCS-docker/tree/master/driver/torcs_client).
