
FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y

RUN apt-get install -y build-essential make cmake ninja-build wget python2.7

ADD download_build_install_llvm.sh /root

RUN mkdir /llvm-3.8.1 && cd /llvm-3.8.1 && /root/download_build_install_llvm.sh