#FROM ubuntu
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

COPY container_init.sh /
COPY run.sh /
COPY pangolin_setup_debugged.py /
COPY ORBvoc.txt.tar.gz /

RUN tar -xvzf ORBvoc.txt.tar.gz && ls -a

RUN apt-get update && apt-get install -y python-software-properties software-properties-common &&  \
    apt-get upgrade -y && apt-get install -y gnupg && \
    add-apt-repository ppa:x2go/stable \
    && add-apt-repository ppa:jonathonf/python-3.6

#RUN apt-get update && apt-get install -y software-properties-common &&  \
#    apt-get upgrade -y && apt-get install -y gnupg && \
#    add-apt-repository ppa:x2go/stable

RUN apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com:80 E1F958385BFE2B6E

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt update
RUN apt install g++-7 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
RUN update-alternatives --config gcc

RUN apt-get update \
    && apt-get install -y python3.6 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    && apt-get install -y python3.6-dev python3-pip \
#    && apt-get install -y libglew-dev libboost-python1.67-dev
    && apt-get install -y libglew-dev libboost-all-dev



#RUN apt-get update \
#    && apt-get install -y python3 \
#    && apt-get install -y python3-dev python3-pip

RUN apt-get update && apt-get install -y less locales sudo zsh x2goserver
RUN echo "deb http://packages.x2go.org/debian stretch extras main\n\
deb-src http://packages.x2go.org/debian stretch extras main" \
> /etc/apt/sources.list.d/x2go.list

RUN sed -i 's/# de_CH.UTF-8 UTF-8/de_CH.UTF-8 UTF-8/' /etc/locale.gen && \
    sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    ln -fs /etc/locale.alias /usr/share/locale/locale.alias && \
    locale-gen && update-locale LANG=en_US.UTF-8
#RUN cp /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
#    echo "Europe/Zurich" >  /etc/timezone

# configure system
RUN sed -i 's/^#X11Forwarding.*/X11Forwarding yes/' /etc/ssh/sshd_config && \
    sed -i "s/Port 22/#Port 22/g" /etc/ssh/sshd_config && \
    echo "Port 2222" >> /etc/ssh/sshd_config && \
    x2godbadmin --createdb

RUN mkdir -p /var/run/sshd

RUN chmod +x /*.sh

# install packages
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y xfce4 epiphany

RUN apt-get update \
    && apt-get install -y \
          file \
          git \
          graphviz \
          libcurl3-dev \
          libfreetype6-dev \
          libgraphviz-dev \
          liblapack-dev \
          libopenblas-dev \
          libpng-dev \
          libxft-dev \
          openjdk-8-jdk \
          swig \
          unzip \
          wget \
          zlib1g-dev \
          cmake \
          libeigen3-dev \
          libsuitesparse-dev \
          qtdeclarative5-dev \
          qt5-qmake \
          zip \
          libjpeg-dev \
          libtiff5-dev \
          libopenexr-dev \
          libgtk2.0-dev \
          pkg-config



# standart python libraries
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install numpy \
    && pip3 install PyOpenGL PyOpenGL_accelerate

#COPY pangolin_setup_debugged.py /
###COPY display_x11_debugged.cpp /
##
## pangolin installation
#RUN pip3 install PyOpenGL PyOpenGL_accelerate \
#    && git clone https://github.com/uoip/pangolin.git \
#    && mv pangolin_setup_debugged.py /pangolin/setup.py \
##    && mv display_x11_debugged.cpp /pangolin/src/display/device/display_x11.cpp \
#    && cd pangolin \
#    && mkdir build \
#    && cd build \
#    && cmake .. \
#    && make -j8 \
#    && cd .. \
#    && python3 setup.py install \
#    && cd / && mkdir data

# set up directories
RUN mkdir /slamdoom
RUN mkdir /slamdoom/tmp
RUN mkdir /slamdoom/libs
RUN mkdir /slamdoom/install
RUN mkdir /slamdoom/install/orbslam2
RUN mkdir /slamdoom/install/opencv3

RUN apt-get install -y nano libglew-dev gdb
RUN apt-get update -y

ADD install/opencv3/install.sh /slamdoom/install/opencv3/install.sh
RUN chmod 777 /slamdoom/install/opencv3/install.sh && /slamdoom/install/opencv3/install.sh /slamdoom/libs python3
#RUN chmod +x /slamdoom/install/opencv3/install.sh && /slamdoom/install/opencv3/install.sh /slamdoom/libs python3

ADD install/orbslam2/install.sh /slamdoom/install/orbslam2/install.sh
ADD install/orbslam2/orbslam2_slamdoom.git.patch /slamdoom/install/orbslam2/orbslam2_slamdoom.git.patch
RUN chmod +x /slamdoom/install/orbslam2/install.sh && /slamdoom/install/orbslam2/install.sh

RUN apt-get install -y libcanberra-gtk-module
RUN pip3 install matplotlib
RUN pip3 install bresenham
RUN pip install numpy --upgrade
RUN pip3 install numpy --upgrade
RUN mkdir /root/.matplotlib && touch /root/.matplotlib/matplotlibrc && echo "backend: Qt5Agg" >> /root/.matplotlib/matplotlibrc

# set up matplotlibrc file so have Qt5Agg backend by default
RUN apt-get install -y

RUN git clone https://github.com/BOpermanis/pyORBSLAM2.git

#RUN cd /pyORBSLAM2/src && ./build.sh
#
#RUN export PYTHONPATH=/orbslam/src/build:$PYTHONPATH >> /root/.bashrc
#
##### upgrading GLX #############
RUN apt-get update
RUN apt-get install -y --fix-missing software-properties-common && \
    add-apt-repository -y ppa:xorg-edgers/ppa && apt-get update  \
    && apt install -y libdrm-dev  libx11-dev python-mako libx11-xcb-dev libxcb-dri2-0-dev mesa-utils\
    libxcb-glx0-dev libxxf86vm-dev libxfixes-dev libxdamage-dev libxext-dev libexpat1-dev flex bison scons meson\
#    libxcb-glx0-dev libxxf86vm-dev libxfixes-dev libxdamage-dev libxext-dev libexpat1-dev flex bison meson\
    && git clone https://gitlab.freedesktop.org/mesa/mesa.git \
    && cd mesa \
    && scons libgl-xlib force_scons=1\
#    && meson libgl-xlib \
    && echo "export LD_LIBRARY_PATH=/mesa/build/linux-x86_64-debug/gallium/targets/libgl-xlib/:$LD_LIBRARY_PATH" >> /root/.bashrc

#EXPOSE 2222
#CMD ["/run.sh"]
