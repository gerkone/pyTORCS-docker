from numpy import pi
from torcs_client.torcs_client import Client

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


if __name__ == "__main__":
    C = Client(port = 3001, maxSteps = 100000, vision = True)

    import sysv_ipc as ipc

    key = 1234
    shm = ipc.SharedMemory(key, flags = 0)

    for step in range(C.maxSteps,0,-1):
        buf = shm.read()
        C.get_servers_input()
        drive_example(C)
        C.respond_to_server()
    C.shutdown()
    shm.detach()
