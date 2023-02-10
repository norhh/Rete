FROM rete/ubuntu-16.04-llvm-3.8.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential \
                    cmake gcovr zlib1g-dev libtinfo-dev \
                    locales \
                    python-pip-whl \
                    python3-pip \
                    python3-setuptools  \
                    && apt-get install -y python3-dev \
                    libboost-filesystem-dev \
                    libboost-program-options-dev \
                    libboost-log-dev \
                    llvm \
                    clang && ln -s /usr/bin/python3 /usr/local/bin/python


ADD CMakeLists.txt /rete/
ADD Config.h.in /rete/
ADD Globals.h.in /rete/
ADD rete /rete/rete
ADD tests /rete/tests
ADD tools /rete/tools
ADD thirdparty /rete/thirdparty
ADD learning /learning

RUN mkdir /rete/build && cd /rete/build \
    && cmake .. -DRETE_LLVM=/llvm-3.8.1 \
    && make

RUN pip3 install clang
ENV Clang_DIR="/llvm-3.8.1/share/clang/cmake/"             
ENV LLVM_DIR="/llvm-3.8.1/share/llvm/cmake/"

ENV PATH="/f1x/build/tools:${PATH}"

CMD ["/bin/bash"]

