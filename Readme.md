# Reproduction Package #

This is the reproduction package for Rete.

## File System Contents ##

* `Rete-Trident` - Source code for running rete + Trident
* `rete-feature-extractor` - Feature extraction process of Rete

## Instructions to Run each of the Components ##

The individual `Readme.md`'s in each directory should discuss how they are run.

### Building the Container

Building the Dockerfile
```
docker build . -t reproduction_package
```

Going into the docker container
```
docker run -v $(pwd):/home/Trident/ --rm -ti reproduction_package /bin/bash
```
### Tools

The package helps in running the following tools
- Rete    -->   Tool constructed in this paper.
- Rete + Trident  --> A combination of Rete and Trident.
- Trident  --> (Trident)[https://ieeexplore.ieee.org/document/9611365]

