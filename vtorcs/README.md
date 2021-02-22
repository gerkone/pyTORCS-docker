# VTORCS: Visual TORCS server with color vision
This is an all in one package of TORCS. Be aware that some included.

## Linux HOST Installation from Source

- Requires plib 1.8.5, FreeGLUT or GLUT, be aware to compile plib with -fPIC
  on AMD64 if you run a 64 bit version of Linux. Be aware that maybe just
  1.8.5 works.
- Untar the archive
- cd into the vtorcs-RL-color directory
- ./configure (use --help for showing the options, of interest might be
  --enable-debug and --disable-xrandr).

```
sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev
./configure
make
make install
make datainstall
```
start with "torcs" (without vision) or "torcs -vision" (with vision)


## Command line arguments:
* -l list the dynamically linked libraries
* -d run under gdb and print stack trace on exit, makes most sense when compile
     with --enable-debug
* -e display the commands to issue when you want to run under gdb
* -s disable multitexturing, important for older graphics cards
* -m use X mouse cursor and do not hide it during races
* -vision use vision input
