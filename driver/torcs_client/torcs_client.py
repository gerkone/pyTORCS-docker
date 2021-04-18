# snakeoil extended with vision through shared memory
# Chris X Edwards <snakeoil@xed.ch>
# Gianluca Galletti


import socket
import sys
import getopt
import numpy as np
import sysv_ipc as ipc

from torcs_client.utils import SimpleLogger as log, reset_torcs, destringify, raw_to_rgb

# max message length
UDP_MSGLEN = 800

SHMKEY = 1234

class Client():
    """
    Snake Oil is a Python library for interfacing with a TORCS
    race car simulator which has been patched with the server
    extentions used in the Simulated Car Racing competitions.
    """
    def __init__(self, host = "localhost", port = 3001, sid="SCR", trackname = None,
                max_steps = 10000, container_id = "0", vision=False, verbose = False,
                img_height= 640, img_width = 480, max_packets = 2):

        self.data_size = 2**17
        # bufsize of the incoming packets (bytes)
        self.max_packets = max_packets
        self.bufsize = self.max_packets * UDP_MSGLEN
        self.verbose = verbose

        self.container_id = container_id

        self.vision = vision

        self.just_started = True

        self.img_height = img_height
        self.img_width = img_width

        self.host = host
        self.port = port
        self.sid = sid
        self.trackname = trackname
        self.max_steps = max_steps  # should be 50steps/second if it had real time performance

        self.reset()

        if(self.vision):
            self.setup_shm_vision()

    def reset(self):
        """
        reset udp connection to vtorcs
        """
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()
        # spin once to avoid empty server state dictionary
        self.get_servers_input()
        self.respond_to_server()

    def shutdown(self):
        if not self.so: return
        if self.verbose: log.alert(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.max_steps,self.port)))
        self.so.close()
        self.so = None

    def restart(self):
        if not self.so: return
        self.so.close()
        self.so = None
        self.reset()

    def setup_connection(self):
        # == Set Up UDP Socket ==
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            if self.verbose: log.error("Could not create socket.")
            sys.exit(-1)
        # Initialize Connection To Server
        self.so.settimeout(1)
        # set socket receive buffer to about 4 packets (maximum observation delay of around 200 ms)
        # this is done to evoid bufferbloat and packet accumulation
        self.so.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.bufsize)
        n_fail = 3
        if self.verbose: log.info("Waiting for server on port {}".format(self.port))
        while True:
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg="{}(init {})".format(self.sid, a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(self.data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error as emsg:
                if n_fail < 0:
                    if self.verbose: log.alert("Could not connect to port {}. Relaunch torcs".format(self.port))
                    reset_torcs(self.container_id, self.vision, True)
                    n_fail = 3
                n_fail -= 1

            identify = "***identified***"
            if identify in sockdata:
                if self.verbose: log.info("Client connected on port {}".format(self.port))
                break

    def setup_shm_vision(self):
        """
        Open shared memory with the key that specified in torcs
        """
        # no need to attach/detach - read only access (nobody likes semaphores)
        self.shm = ipc.SharedMemory(SHMKEY, flags = 0)

    def get_vision(self):
        """
        Return a numpy array with the whole 640x480(x3) vision
        """
        if(self.vision):
            # the first image in the shared memory is the last image of the previous run
            # skip it
            if(hasattr(self, "shm") and self.just_started):
                # read image size, 16 padding pits should be there otherwise
                buf = self.shm.read(self.img_width * self.img_height * 3)
                # sent as array of 8 bit ints
                image_buf = np.frombuffer(buf, dtype=np.int8)

                return raw_to_rgb(image_buf, self.img_width, self.img_height)
            else:
                # shared memory not yet ready, get blank image
                image_buf = np.zeros(self.img_width * self.img_height * 3)
                return raw_to_rgb(image_buf, self.img_width, self.img_height)

        return None

    def get_servers_input(self):
        """
        Server"s input is stored in a ServerState object
        """
        if not self.so: return
        sockdata= str()

        while True:
            try:
                # Receive server data
                sockdata, addr= self.so.recvfrom(self.data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error as emsg:
                log.error("Socket error raised: {}".format(emsg))
            if "***identified***" in sockdata:
                if self.verbose: log.info("Client connected on port {}".format(self.port))
                continue
            elif "***shutdown***" in sockdata:
                if self.verbose: log.alert("Server has stopped the race on {}. You were in {} place".format(self.port, self.S.d["racePos"]))
                self.shutdown()
                return
            elif "***restart***" in sockdata:
                if self.verbose: log.alert("Server has restarted the race on port {}.".format(self.port))
                # reset UDP, reset client
                self.restart()
                return
            elif not sockdata: # Empty?
                continue       # Try again.
            else:
                self.S.parse_server_str(sockdata, self.get_vision())
                break # Can now return from this function.

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            log.error("Error sending to server: {} Message {}".format(emsg[1], str(emsg[0])))
            sys.exit(-1)

class ServerState():
    """
    What the server is reporting right now.
    """
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string, image):
        """
        Parse the server string.
        """
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip("(").rstrip(")").split(")(")
        for i in sslisted:
            w= i.split(" ")
            self.d[w[0]]= destringify(w[1:])

        if(image is not None):
            # add image to the state dictionary
            self.d["img"] = image

class DriverAction():
    """
    What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)
    """
    def __init__(self):
       self.actionstr= str()
       # "d" is for data dictionary.
       self.d= { "accel":0.2,
                   "brake":0,
                  "clutch":0,
                    "gear":1,
                   "steer":0,
                   "focus":[-90,-45,0,45,90],
                    "meta":0
                    }

    def clip_to_limits(self):
        self.d["steer"] = np.clip(self.d["steer"], -1, 1)
        self.d["brake"] = np.clip(self.d["brake"], 0, 1)
        self.d["accel"] = np.clip(self.d["accel"], 0, 1)
        self.d["clutch"] = np.clip(self.d["clutch"], 0, 1)
        if self.d["gear"] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d["gear"]= 0
        if self.d["meta"] not in [0,1]:
            self.d["meta"]= 0
        if type(self.d["focus"]) is not list or min(self.d["focus"])<-180 or max(self.d["focus"])>180:
            self.d["focus"]= 0

    def __repr__(self):
        self.clip_to_limits()
        out= str()
        for k in self.d:
            out+= "("+k+" "
            v= self.d[k]
            if not type(v) is list:
                out+= "%.3f" % v
            else:
                out+= " ".join([str(x) for x in v])
            out+= ")"
        return out
