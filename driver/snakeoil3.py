from numpy import pi

from torcs_client.torcs_client import Client
from torcs_client.utils import start_container, reset_torcs

def drive_example(c):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    S,R= c.S.d,c.R.d
    target_speed=100
    # Steer To Corner
    R['steer']= S['angle']*10 / pi
    # Steer To Center
    R['steer']-= S['trackPos']*.10

    # Throttle Control
    if S['speedX'] < target_speed - (R['steer']*50):
        R['accel']+= .01
    else:
        R['accel']-= .01
    if S['speedX']<10:
       R['accel']+= 1/(S['speedX']+.1)

    # Traction Control System
    if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
       (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 5):
       R['accel']-= .2

    # Automatic Transmission
    R['gear']=1
    if S['speedX']>50:
        R['gear']=2
    if S['speedX']>80:
        R['gear']=3
    if S['speedX']>110:
        R['gear']=4
    if S['speedX']>140:
        R['gear']=5
    if S['speedX']>170:
        R['gear']=6
    return

def main(verbose = False, hyperparams = None, sensors = None, image_name = "gerkone/torcs", img_width = 640, img_height = 480):
    container_id = start_container(image_name, False, 3001)
    reset_torcs(container_id, True, kill = True)
    C = Client(verbose = verbose, port = 3001, maxSteps = 100000, vision = True, container_id = container_id)

    for step in range(C.maxSteps,0,-1):
        C.get_servers_input()
        drive_example(C)
        C.respond_to_server()
    C.shutdown()


if __name__ == "__main__":
    main()
