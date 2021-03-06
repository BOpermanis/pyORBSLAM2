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
    add-apt-repository ppa:x2go/stable
RUN apt-key adv --recv-keys --keyserver hkp://keyserver.ubuntu.com:80 E1F958385BFE2B6E

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

RUN apt install -y libpcl-dev

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
          python3-dev \
          python3-pip \
          python3 \
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
RUN pip3 install pip==19 \
    && python3 -m pip install numpy
#    && python3 -m pip install opencv-python==3.3.1.11 \
#    && python3 -m pip install opencv-contrib-python==3.3.1.11

RUN apt-get install -y libboost-all-dev

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt update
RUN apt install g++-7 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
RUN update-alternatives --config gcc

RUN git clone https://github.com/BOpermanis/pyORBSLAM2.git

RUN apt-get install -y libglew-dev

COPY pangolin_setup_debugged.py /
##COPY display_x11_debugged.cpp /
#
# pangolin installation
RUN pip3 install PyOpenGL PyOpenGL_accelerate \
    && git clone https://github.com/uoip/pangolin.git \
    && mv pangolin_setup_debugged.py /pangolin/setup.py \
#    && mv display_x11_debugged.cpp /pangolin/src/display/device/display_x11.cpp \
    && cd pangolin \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j8 \
    && cd .. \
    && python3 setup.py install \
    && cd / && mkdir data

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

#RUN cd /pyORBSLAM2/src && ./build.sh

RUN export PYTHONPATH=/orbslam/src/build:$PYTHONPATH >> /root/.bashrc

##### upgrading GLX #############
RUN apt-get update
RUN apt-get install -y --fix-missing software-properties-common && \
    add-apt-repository -y ppa:xorg-edgers/ppa && apt-get update  \
    && apt install -y libdrm-dev  libx11-dev python-mako libx11-xcb-dev libxcb-dri2-0-dev mesa-utils\
    libxcb-glx0-dev libxxf86vm-dev libxfixes-dev libxdamage-dev libxext-dev libexpat1-dev flex bison scons meson \
    && git clone https://gitlab.freedesktop.org/mesa/mesa.git
RUN cd mesa \
    && git checkout 159abd527ec191e8274876162b30079c4ea39600 \
    && scons libgl-xlib force_scons=1\
#    && meson libgl-xlib \
    && echo "export LD_LIBRARY_PATH=/mesa/build/linux-x86_64-debug/gallium/targets/libgl-xlib/:$LD_LIBRARY_PATH" >> /root/.bashrc


RUN apt install -y gedit

RUN git clone https://github.com/IntelRealSense/librealsense.git \
    && apt-get install -y libssl-dev libusb-1.0-0-dev libgtk-3-dev libglfw3-dev \
    && cd librealsense && ./scripts/setup_udev_rules.sh \
    && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make

RUN cd /librealsense/build && make install

RUN pip3 install pillow && pip3 install scipy && pip3 install sklearn && pip3 install pyrealsense2 && pip3 install open3d

RUN echo "alias python=python3" >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH="/usr/local/lib"' >> /root/.bashrc


# installing TEASER-plusplus
COPY cmake-3.20.2-linux-x86_64.sh /
COPY install_cmake.sh /
RUN chmod +x cmake-3.20.2-linux-x86_64.sh
RUN sh -c "/bin/echo -e 'y' | bash install_cmake.sh"
RUN echo "export PATH='/cmake-3.20.2-linux-x86_64/bin/:$PATH'" >> /root/.bashrc


COPY eigen-3.2.10.zip /
RUN unzip eigen-3.2.10.zip
RUN cd eigen-3.2.10 && mkdir build && cd build && /cmake-3.20.2-linux-x86_64/bin/cmake .. && make && make install
COPY Eigen3Config.cmake /eigen-3.2.10/build

COPY Miniconda3-py39_4.9.2-Linux-x86_64.sh /

RUN git clone https://github.com/BOpermanis/TEASER-plusplus.git

RUN cd TEASER-plusplus \
    && mkdir build \
    && cd build \
    && /cmake-3.20.2-linux-x86_64/bin/cmake -DTEASERPP_PYTHON_VERSION=3.5 .. \
    && make \
    && sudo make install \
    && cd .. && cd examples/teaser_cpp_ply \
    && mkdir build && cd build && cmake .. && make

EXPOSE 2222
CMD ["/run.sh"]