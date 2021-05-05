# Deep sensor estimation and DDPG
Normal DDPG algorithm, but input data is estimated from a stack of 3 frames.

The estimated values are:
- track - 19 rangefinder values, fired from the front of the car in the range [-45°,45°]
- angle - angle with the center line
- speedX - transversal speed
- trackPos - distance from center line

The tested estimator works with a shallow ResNet.

Note that this is a very simple experiment. The network works very good only on _g-track-1_ (>90% accuracy) but not so good on other tracks (~50-60%).
