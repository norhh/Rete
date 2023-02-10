# Installation
Build the docker container in `infra`, and then build the container in the current folder.

```
> cd infra/
> docker build -t rete/ubuntu-16.04-llvm-3.8.1 . 
> cd ..
> docker build -t rete-feature .
```

# Running
Run the docker container by first mounting your current directory into `/tmp` to `/bin/bash`
```
> docker run -v $(pwd):/tmp --rm -it rete /bin/bash
```

You can ignore the mounting if you do not need the current directory. You can either compile inside docker
after mounting or directly run the build code present in `/rete` folder.

```
> cd /tmp/rete
> cmake .. -DF1X_LLVM=/llvm-3.8.1
> make
> chmod u+x rete
```

To extract a json of feature information.
```
> ./rete  -output="<path>"
```

To extract intermediate CDU chain data.
```
> ./rete -get-chain-data -output="<path>"
```

