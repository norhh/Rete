#!/bin/sh
apt-get update

apt-get install -y build-essential libapache2-mod-gnutls git asciidoc libgcrypt11-dev libgtk-3-dev qttools5-dev-tools qtdeclarative5-dev qml-module-qtquick-controls libtool-bin automake autoconf libtool subversion pkg-config bison flex libgtk2.0-dev libpcap-dev libwww-perl
apt-get install -y qt5-default qtscript5-dev libssl-dev qttools5-dev qttools5-dev-tools qtmultimedia5-dev libqt5svg5-dev libqt5webkit5-dev libsdl2-dev libasound2 libxmu-dev libxi-dev freeglut3-dev libasound2-dev libjack-jackd2-dev libxrandr-dev libqt5xmlpatterns5-dev libqt5xmlpatterns5 libqt5xmlpatterns5-private-dev
# Basically will install any more missing deps
apt-get install -y wireshark