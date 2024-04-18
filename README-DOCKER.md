# Landshark using docker
Install docker in your computer.

Then use the following command to build your docker image:

```bash
$ cd /path/to/landshark_repo_in_your_pc/ 
$ docker build -t landshark:0.1 .
```

Once the build is successful, make sure the image is available in you pc using:

```bash
$ docker images
``` 

Finally, run your docker image:

```bash
$ docker run -it -v /path/to/landshark_clone_in_your_pc:/usr/src/landshark/ --restart=always landshark:0.1 /bin/bash
```

This will take you inside the docker container on a bash shell. Install `landshark` in your docker container:

```bash
$ pip install -e .[dev] 
```

Check the tests can be run:
```bash
$ pytest tests -sx
```
