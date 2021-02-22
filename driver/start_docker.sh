nvidia-docker run -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY -p 3101:3101/udp -it "$1"
