# VTORCS: Visual TORCS server with color vision
This is an all in one package of TORCS.

## Linux HOST Installation from Source
TORCS can also be eventually installed on the host.

- Requires plib 1.8.5, FreeGLUT or GLUT, be aware to compile plib with -fPIC
  on AMD64 if you run a 64 bit version of Linux. Be aware that maybe just
  1.8.5 works.
- cd into the vtorcs directory
- ./configure (use --help for showing the options, of interest might be
  --enable-debug and --disable-xrandr).

```
sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev
./configure
make
make install
make datainstall
```
Verify that the installation works by running 

```
torcs -vision
```

and selecting play > practice > start. If it stops on a blue screen with "waiting for scr_server" it works properly.

## Configuration
**important** : If you dont follow this the visual input won't work properly.

After the manual installation run postinstall.sh. 
```
sh postinstall.sh
```

This will copy the right TORCS configuration files. Tis includes:
* disabled audio
* correct resolution (the game needs to be run at 64x64 or the vision does not work properly)
* bumper view of the car
* a simple track


## Command line arguments:
* -l list the dynamically linked libraries
* -d run under gdb and print stack trace on exit, makes most sense when compile
     with --enable-debug
* -e display the commands to issue when you want to run under gdb
* -s disable multitexturing, important for older graphics cards
* -m use X mouse cursor and do not hide it during races
* -vision use vision input
