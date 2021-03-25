# **TORCS 1.3.7** 
Version of TORCS 1.3.7 with [SCR patch](https://github.com/barisdemirdelen/scr-torcs-1.3.7) and an additional patch to send the current game image to another application via shared memory.

## Installation on Ubuntu 20.04

For Ubuntu 20.04, please proceed as follow:

### Install all necessary requirements

```
sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng-dev libvorbis-dev
```

### Build torcs

```
$ export CFLAGS="-fPIC"
$ export CPPFLAGS=$CFLAGS
$ export CXXFLAGS=$CFLAGS
$ ./configure --prefix=$(pwd)/BUILD  # local install dir
$ make
$ make install
$ make datainstall
```
### Run torcs

To run torcs with local installation, execute

```
./your_path_to_torcs/torcs-1.3.7/BUILD/bin/torcs
```

## Installation on Ubuntu 18.04

For Ubuntu 18.04, please proceed as follow:

### Install all necessary requirements

```
sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng-dev
```

### Build torcs

```
$ export CFLAGS="-fPIC"
$ export CPPFLAGS=$CFLAGS
$ export CXXFLAGS=$CFLAGS
$ ./configure --prefix=$(pwd)/BUILD  # local install dir
$ make
$ make install
$ make datainstall
```

### Usage with ROS

If you want to run this software with the ROS adapter, you also need to install opencv uing `$apt-get install -y *opencv*`, after installing the ROS Melodic release.

## Installation on Ubuntu 16.04

### install torcs dependencies
first we need to get some necessary debian packages

```sudo apt-get install mesa-utils libalut-dev libvorbis-dev cmake libxrender-dev libxrender1 libxrandr-dev zlib1g-dev libpng16-dev```

now check for openGL/DRI by running

```glxinfo | grep direct```

the result should look like

```direct rendering: Yes```

check for glut by running

```dpkg -l | grep glut```

if it is not installed run

```sudo apt-get install freeglut3 freeglut3-dev```

check for libpng by running

```dpkg -l | grep png```


#### install PLIB

first we have to create a folder for all torcs-related stuff. Therefore, run the following commands

```cd /your_desired_location/```

```sudo mkdir torcs```

```export TORCS_PATH=/your_desired_location/torcs```

```cd $TORCS_PATH```

install PLIB-dependencies

```sudo apt-get install libxmu-dev libxmu6 libxi-dev```

now download [PLIB 1.8.5](http://plib.sourceforge.net/download.html), unpack to the created directory and enter the plib folder by 

```sudo tar xfvz /path_to_downloaded_files/plib-1.8.5.tar.gz```

```cd plib-1.8.5```

before we compile plib we need need to set some environment variables

```export CFLAGS="-fPIC"```

```export CPPFLAGS=$CFLAGS```

```export CXXFLAGS=$CFLAGS```

now we can configure and compile PLIB

```./configure```

```make```

```sudo make install```

just for safety, wen unset our environment variables again

```export CFLAGS=```

```export CPPFLAGS=```

```export CXXFLAGS=```

#### install openal
let's enter our base directory again

```cd $TORCS_PATH```

now we download [openal 1.17.2](http://kcat.strangesoft.net/openal-releases/) and unpack it

```sudo tar xfvj /path_to_downloaded_files/openal-soft-1.17.2.tar.bz2 ```

we enter the build folder and compile openal

```cd openal-soft-1.17.2/build```

```sudo cmake ..```

```sudo make```

```sudo make install```

### install TORCS
enter your TORCS_PATH 

```cd $TORCS_PATH```

and clone this repository

```git clone https://github.com/fmirus/torcs-1.3.7.git```

now we enter our torcs folder 

```cd torcs-1.3.7```

now build we build TORCS and log the output to a text-files as TORCS does not interrupt the build on errors

```make >& error.log```

now open error.log with your favourite text editor and search for errors. If there are no errors you can proceed, otherwise you have to resolve.

now we are ready to install torcs by running

```sudo make install```

also install the torcs data-files by running

```sudo make datainstall```

If you made it this far, you can delete the TORCS_PATH variable by ```unset TORCS_PATH``` and are now ready to go. Congratulations :-)

# Original TORCS README:

1.  Introduction
2.  Documentation
3.  Non-Free content (in GPL sense)
4.  Track editor
5.  Linux Installation from Source
6.  Windows Installation from Source (Release version)
6.1 Windows Installation from Source, additional notes
7.  Windows Installation from Source (Debug version)
8.  Testing
9.  Getting Help
10. Running under Valgrind with Linux
11. Changes
12. TODO/Notes


## 1. Introduction
---------------
First a big welcome, I hope you will enjoy your ride:-)

This is an all in one package of TORCS. Be aware that some included
artwork has non free (in the GPL sense) licenses, you will find a "readme.txt"
in those directories. The rest is either licensed under the GPL or the Free
Art License. If you want to create cars or advanced tracks using the accc tool,
you will require stripe from http://www.cs.sunysb.edu/~stripe.

If you use TORCS for research/projects you can have a look into the FAQ for
citation guidelines.

Kind regards

Bernhard


## 2. Documentation
----------------
You can find a variety of links on www.torcs.org (video tutorials about content
creation/usage, written documentation like the robot tutorial, etc.). The TORCS
API and architecture documentation can be generated with Doxygen 1.8, run
"make doc", the result can be found in doc/manual/api, point to index.html.


## 3. Non-Free content (in GPL sense)
----------------------------------
Here the list with the directories containing non free content, look at the
readme.txt for details:
- data/cars/models/pw-*
- data/cars/models/kc-*


## 4. Track editor
---------------
The track editor is not included in this distribution, you can get it from
http://www.berniw.org/trb/download/trackeditor-0.6.2c.tar.bz2, the sources
are included in the jar. The sources are also available here:
http://sourceforge.net/projects/trackeditor.


## 5. Linux Installation from Source
---------------------------------
- Requires plib 1.8.5, FreeGLUT or GLUT, be aware to compile plib with -fPIC
  on AMD64 if you run a 64 bit version of Linux. Be aware that maybe just
  1.8.5 works.
- Untar the archive
- cd into the torcs-1.3.5 directory
- ./configure (use --help for showing the options, of interest might be
  --enable-debug and --disable-xrandr).
- make
- make install
- make datainstall
- start with "torcs"

Command line arguments:
* -l list the dynamically linked libraries
* -d run under gdb and print stack trace on exit, makes most sense when compiled
     with --enable-debug
* -g run under Valgrind (requires a debug build for useful results)
* -e display the commands to issue when you want to run under gdb
* -s disable multitexturing, important for older graphics cards
* -m use X mouse cursor and do not hide it during races
* -r pathtoraceconfigfile, run race from command line only, for testing and AI
     training, see FAQ for details
* -k (keep) suppress calls to dlclose to keep modules loaded (for Valgrind runs,
     to avoid "??" in the call stack)


## 6. Windows Installation from Source (Release version)
-----------------------------------------------------
- hint: you can have a release and a debug build side by side, the release
  version goes to "runtime" and the debug to "runtimed".
- requires VS 6 (tested with sp6) or VS 2008 (tested with sp1), VS2010 is reported
  to work as well. For express editions or VS 2012 read notes in section 6.1.
- VS 6.0 support is fading out, you will need to install the Windows Server 2003
  February Edtion CORE SDK (the last one which worked with VS 6.0) and set the lib
  and include path in the options (used for SHGetFolderPath etc.).
- untar the archive into a path without whitespaces and special characters.
- cd into the torcs-1.3.5 directory
- run setup_win32.bat
- run setup_win32-data-from-CVS.bat
- select the TORCS workspace (TORCS.dsw for VS 6) or solution (TORCS.sln
  for VS 2008), select the w32-Release version.
- compile project (0 warnings)
- cd into the "runtime" directory.
- run "wtorcs.exe"

Command line arguments:
* -s disable multitexturing, important for older graphics cards
* -r pathtoraceconfigfile, run race from command line only, for testing and AI
     training, see FAQ for details 


### 6.1 Windows Installation from Source, additional notes
------------------------------------------------------
#### 6.1.1 VS 2005 Express (based on imported dsw), reported by Eric Espie:
- Run up to the setup*.bat step in the above instructions, then open the TORCS.dsw
  file and do the following changes
- in wtorcs -> Source Files (Solution explorer) exclude torcs.rc
- in client -> Source Files add the file errno.cpp to the solution (located
  in src/libs/client)
- change in the properties of all the sub-projects :
        in "Configuration Properties -> Link Editor -> Entry : Ignore Specific Library"
        change LIBCD in LIBC if present.

#### 6.1.2 VS 2005 Express (based on VS2008 sln), reported by Wolf-Dieter Beelitz:
- Edit all vcproj (=xml) files and set the "version" from 9.00 to 8.00
- Follow the instructions above.

#### 6.1.3 VS 2008 Express, reported by Stacey Pritchett:
- in wtorcs -> Source Files (Solution explorer) exclude torcs.rc
- Follow the instructions above.

#### 6.1.4 VS 2012, reported by SteveO:
- In every project (except TORCS) add /SAFESEH:NO into the Additional Options
  (Properties-Configuration Properties-Linker-Command Line), see also
  http://msdn.microsoft.com/en-us/library/9a89h429.aspx.


## 7. Windows Installation from Source (Debug version)
---------------------------------------------------
- hint: you can have a debug and a release build side by side, the debug
  version goes to "runtimed" and the release to "runtime".
- requires VS 6 (tested with sp6) or VS 2008 (tested with sp1), VS2010 is reported
  to work as well. For express editions or VS 2012 read notes in section 6.1.
- VS 6.0 support is fading out, you will need to install the Windows Server 2003
  February Edtion CORE SDK (the last one which worked with VS 6.0) and set the lib
  and include path in the options (used for SHGetFolderPath etc.).
- untar the archive into a path without whitespaces and special characters.
- cd into the torcs-1.3.5 directory
- run setup_win32_debug.bat
- run setup_win32-data-from-CVS_debug.bat
- select the TORCS workspace (TORCS.dsw for VS 6) or solution (TORCS.sln
  for VS 2008), select the w32-Debug version
- compile project (0 warnings)
- cd into the "runtimed" directory.
- run "wtorcs.exe"

Command line arguments:
* -s disable multitexturing, important for older graphics cards
* -r pathtoraceconfigfile, run race from command line only, for testing and AI
     training, see FAQ for details 


## 8. Testing
----------
If you find problems which should be already fixed or new ones please report them
to the torcs-users mailing list.


## 9. Getting Help
---------------
During the game press F1. For more in depth information visit www.torcs.org,
you find there a lot of information, look at the documentation section on
the left, have as well a look into the list of howto's. If you are stuck
have a look into the FAQ to learn how and where to report a problem.


## 10. Running under Valgrind with Linux
------------------------------------
First you need to build a debug version of TORCS, make sure that the CFLAGS,
CPPFLAGS and CXXFLAGS environment variables are empty (usually they are). Then
run "make distclean", then the configure script with the option --enable-debug
and all other options which you require, build and install as usual.

To find memory leaks run first (Valgrind must be available in the path):
./torcs -g

You will find the logfile valgrind.log in the .torcs directory. If you have
"??" in the call stack, you can run TORCS with the -k option to avoid unloading
the modules:
./torcs -g -k

You should use -k just to investigate the "??" in the call stacks, because the
suppression of dlclose can hide problems related with module release and cause
problems because modules are just recycled but not reloaded.

Of course you can use this with the console (command line) mode as well, e.g.:
./torcs -g -r ~/.torcs/config/raceman/champ.xml
./torcs -g -k -r ~/.torcs/config/raceman/dtmrace.xml

Some additional notes:
- Valgrind (version 3.6.1) reports on systems with the ATI flgrx OpenGL driver (8.961)
  lots of leaks, according AMD Valgrind misinterprets memory blocks handed over to the
  kernel. When I wrote suppressions the flgrx driver hung the X Server up, conclusion:
  Give it a try (maybe another Valgrind/driver/kernel combination does/will do better),
  but if you run in the mentioned problems, just use the TORCS command line mode or
  install temporarily the Open Source ATI driver alternative, maybe this does do better
  (not tested, send me a report;-) )
- You can edit the "torcs" script and add "--leak-check=full --show-reachable=yes"
  to see what is still reachable at exit. This is useful to reduce the amount of cached
  xml file handles or hunt down missing releases of handles (they are not reported
  usually because they are reachable via the cache), beware, it is perfectly
  fine that the GUI and some handles are held permanent.


## 11. Changes
-----------

Changes since 1.3.6
-------------------
- Added missing pictures for Doxygen generated documentation (Bernhard).
- Fixed all Doxygen (version 1.8.2) warnings (Bernhard).
- Adjusted Doxygen configuration (Bernhard).
- Added architecture overview to documentation (Bernhard).
- Updated documentation in params.cpp (Bernhard).
- Improved some currently unused functions in params.cpp (Bernhard).
- params.cpp cleanup (Bernhard).
- Restructured/improved robottools documentation (Bernhard).
- Improved documentation of interfaces (track, graphic, robot, simu) (Bernhard).
- Improved pointer checking in RmLoadingScreenSetText (Bernhard).
- Improved race manager API documentation (Bernhard).
- Added RmGetCategoryName to race manager API, as the name suggests (Bernhard).
- Removed obsolete file confscreens.h (Bernhard).
- Enabled documentation client side (Javascript) search engine (Bernhard).
- Improved ReApplyRaceTimePenalties for cases where drivers did not complete a
  single lap or the car has been wrecked (Bernhard).
- Removed some outdated files from human driver (Bernhard).
- TORCS configuration and result files go now to correct place on Windows,
  e.g. to AppData/Local/torcs on Windows 7 (Bernhard).
- Result saving creates directory if not available, matters when creating
  custom racemanagers or running custom batches with -r (Bernhard).
- Improved -r on Windows, paths containing backslashes ('\') are now working
  (Bernhard). 
- Added ShFolder.lib to VS 6 project files (required for SHGetFolderPath).
  For VS 6 builds you will need to install the Windows Server 2003 February
  Edtion CORE SDK (the last one which worked with VS 6) and set the lib
  and include path in the VS 6 options (Bernhard).
- Fixed some gcc 4.8.1 warnings (Bernhard).
- Disabled penalties after race finish, reported by MarkP (MarkP, Bernhard).
- Added new options to trackgen for testing, see -i, -o (Bernhard).
- Added a testsuite to generate a bulk of tracks ("test" directory) (Bernhard).
- Added blacklisting of button events in player perferences, needed to set up
  input devices which fire button and axis events on analogue buttons, e.g.
  L2/R2 of playstation 4 controllers. Add in the drivers section of
  preferences.xml e.g. <attstr name="blacklisted events" val="BTN7-0,BTN8-0"/>
  (Bernhard).
- Improved wheel velocity calculation (Bernhard).
- Improved suspension code to catch damping spikes in extreme conditions and
  setups, for TRB (Wolf-Dieter, Bernhard).
- Added comments in susp.cpp (Bernhard).
- Ensure that the third element just produces positive forces (Wolf-Dieter,
  Bernhard).

## 12. TODO/Notes
--------------

TODO for 1.3.8 "worn & blown"
--------------
- Z-Collision
- Eventually Z calculation
- Caster
- Threshold and caster adjustable- Dynamic track
- Wind/Temp
- Tire Wear/Temp
- Curb sound
- Brake balance adjustable during ride
- Eventually differential(s) adjustable during ride

TODO for 1.3.9 "analysed"
--------------
- Data recorder ("Telemetry")
- Data analyser (high/low-pass filtering, comparison, etc.)
- Speed/Shock
- Replay?

TODO for 1.3.10 "ruled"
---------------
- Rules
- Timed races (e.g. 24h).
- Rules/Modes which do not requrie 1.4 changes

TODO for 1.3.11 "tutored"
---------------
- Starting/race modes (multi-class for TRB?)
- VS update
- Robot Tutorial update

TODO for 1.3.12 "managed"
---------------
- Document race manager XML
- Expose secret settings in GUI (e.g. Button masking)

TODO for 1.3.13 or 1.4.x "artistic"
------------------------
- Ingame track generation wizard (themed)
- Ingame car livery design (themed)

TODO for 1.3.14 or 1.4.x "everywhere"
------------------------
- Merge simuv3 parts into simuv2
- Review and eventually apply mac os x build
- Review MorphOS changes
- Review WD's ABS suggestion, apply.
- Review WD's 4WD analysis/patch.

TODO for 1.3.15+
----------------
- Maintainance only, move development to 1.4
- Compiler and library adoptions, fixes, compatible content updates


TODO for 1.4.x
--------------
- Robot interface adoptions (maybe askfeatures, callonce, grid, postprocess, we will see...)
- More Rules.
- Brake/Engine Wear (with cooling)
- Apply SDL patch from Brian Gavin but moving directly to SDL 2.0.
- Update OpenAL to 1.1, fix use of deprecated stuff.
- Car asignment for human players.
- Sound (alut depricated/scheduling for lots of cars)
- Make it possible for a robot module to carry any arbitrary number of
  drivers (instead of static 10).
- Pace car and trucks to remove wrecks (remove "virtual" crane?).
- Replace wav sounds with ogg?
- Track extensions (crossings, split/join, variable width)
- Replays
- Skidmarks/shadows masking with stencil
- Phong specular highlights/in shadow occlusion
- Skidmarks to simu/persistency (get rid of frame rate dependency)
- Review/reduce dynamic memory allocation/release during rendering
- Store all graphics engine state in a context struct/object (to be able to render telemetry
  in the car setup screen or during a running session)
- track wall properties
- Solve problems with side entering/exiting pit lane rules (repeated violations give
  only one penalty under some conditions).

TODO TRB
--------
- RSS feed(s) (suggested by Quinten)
- Race XML generation
- Content exchange
- E-Mail exchange

TODO for Compliance
-------------------
155-DTM -> replace with car10-trb1
acura -> replace with car9-trb1
mc-larenf1 -> replace with car10-trb1
p406 -> replace with car1-trb4
rework buggy, baja bug
replace rally cars
Remove invalid geometry from tracks 
convert force units internally from lbs to lbf


Later:
-------------------
- Decide about plib (not maintained anymore?) -> integration of minimal subset as base
  for own engine?
- Refactor trackgen (left/right -> half the code, maybe more)
- GUI for event blacklisting.
- Ongoing for every release: rework free car models (several holes, no
  emmissive color of lod (model becomes dark at a certain distance), single
  sided, add cockpit, lods).
- Ongoing for every release: Improve visual quality of some existing tracks.
- Fix sound in split screen multiplayer.
- Ongoing: Replace some defines with "static const" to be able to see the
  symbol name when debugging.
- move berniw/bt spline code into math to share it.
- hunt down all glGet* crap during the simulation.
- (Problem when driver list changes during championship.)
- (add proper init/shutdown to every module which is not bound to anything else
  but the loading/init/shutdown/unloading process.)
- Blind mode should not load graphics engine.


TODO for 1.9.x (pre 2.0 series, no release)
--------------
- Design networking, how to embed it into TORCS?
- Networking prototype.
- Gaming modes suitable for online races.
- Cockpit inside view.
- Set up infrastructure for reading binary data files bit with and endianness independent.

TODO for 2.0.0
--------------
- Initial Networking.


TODO LATER
----------
- Add validation for the case no driver selected, do not exit to
  console.
- Networking (2.0).
- SMP simulaton core (for myself).
- Replays.
- Telemetry recorder/viewer.
- Phong specular highlights (optional env, cube or GLSL).
- Shadowmapped/Stenciled dynamic car shadows.
- so/dll with libtool, common code?
- 3d-grass.
- Dynamic sky.
- TRB integration.
- Show just fitting resolutions for fullscreen/change for windowed mode.
- Separate components more clean (e.g. ssgInit should go back to
  ssggraph, etc.)
- Avoid cameras cutting the landscape.
- Start position marks on track (same technique like fake shadow, skids).
- Start procedures (pace car, etc).
- Better transparency for track objects.
- More driving aids, switch to AI and back.
- Opponent sets for human players (e.g 20 Open Wheel cars, etc.)
- Free camera controlled with mouse and keys.


IDEAS FOR MUCH LATER
--------------------
- Weather.
- Dynamic day/night-time, car lights.
- Pit crew.
- Dynamic "intelligent" Objects (e.g. Helicopter)
- Solid/dynamic obstacles.
- Nicer trees etc, terrain LOD.
- Inside view.
- Animated driver.
- Dirt on cars, inside view.
- free terrain.
- Open track dynamically generated when driving.
- Random track generator.
- Separate pit path, Y segments, etc?
- TORCS as benchmark or screensaver?
- Force feedback.
- Story mode with message.
- Traffic simulator
