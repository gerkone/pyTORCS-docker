FROM nvidia/cudagl:11.0-base

# build stuff
RUN apt-get update && apt-get install -y --no-install-recommends make g++ xautomation

# torcs dependencies
RUN apt-get install -y --no-install-recommends libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libplib-dev libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev libxrandr-dev libpng-dev libxxf86vm-dev

RUN rm -rf /var/lib/apt/lists/*

RUN touch ~/.Xauthority

# build and install modified torcs
COPY . vtorcs

WORKDIR vtorcs

RUN chmod +x configure && ./configure
ENV CFLAGS="-fPIC"
ENV CPPFLAGS=$CFLAGS
ENV CXXFLAGS=$CFLAGS
RUN ls src && make
RUN make install && make datainstall

# copy custom configure files
RUN rm -rf /usr/local/share/games/torcs/config
RUN cp -r config /usr/local/share/games/torcs/config

COPY start.sh start.sh
COPY kill.sh kill.sh

# remove useless tools
RUN apt-get remove -y make g++

# open scr server udp port
EXPOSE 3101

CMD ["/bin/bash"]
