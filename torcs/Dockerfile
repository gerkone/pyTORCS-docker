FROM nvidia/cudagl:11.0-base

USER root

# build stuff
RUN apt-get update -y && apt-get install -y --no-install-recommends make g++

# torcs dependencies
RUN apt-get install -y --no-install-recommends libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev \
   freeglut3-dev libplib-dev libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev \
   libxrandr-dev libpng-dev libxxf86vm-dev libvorbis-dev

RUN rm -rf /var/lib/apt/lists/*

RUN touch ~/.Xauthority

# build and install modified torcs
COPY . torcs/

WORKDIR torcs

RUN sh configure
ENV CFLAGS="-fPIC"
ENV CPPFLAGS=$CFLAGS
ENV CXXFLAGS=$CFLAGS
RUN make
RUN make install && make datainstall

# copy custom configure files (remove old stuff first)
RUN rm -rf /usr/local/share/games/torcs/config
RUN rm -rf /usr/local/share/games/torcs/drivers/human
RUN mkdir -p /root/.torcs/
RUN cp -R configs/* /usr/local/share/games/torcs
RUN cp -R configs/* /root/.torcs

# # remove build tools
# RUN apt-get remove -y make g++

RUN cd .. && rm -rf torcs

CMD ["/bin/bash"]
