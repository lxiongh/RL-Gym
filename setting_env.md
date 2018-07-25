## Overview

This project demonstrates how to use Reinforcement Learning Algorithms (DQN, DDPG, etc.) with Pytorch to play games. 


## Installation Dependencies

System: Ubuntu 16.04, 4vCPU, 8G, 2.5GHz, Aliyun ECS


### Pip3

```
apt-get update
apt-get install python3-pip
```

### Pytorch, Gym

``` bash
pip3 install torch torchvision
pip3 install gym_ple pygame
apt-get install -y python-pygame
```

> https://github.com/lusob/gym-ple
> gym_ple requires PLE, to install PLE clone the repo and install with pip.

``` bash
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
```

**Box2D**

``` bash
apt-get install swig git
git clone https://github.com/pybox2d/pybox2d.git
cd pybox2d
python setup.py clean
python setup.py install
```

### Xvfb (Fake screen)

> xvfb should be installed when using linux server (env.render()) 

HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s "-screen 0 1400x900x24" python <your_script.py>'

``` bash
apt-get install xvfb, python-opengl
apt-get install libav-tools
```

### Jupyter

> It is easy to coding in Jupyter Web UI http://0.0.0.0:8888/.

``` bash
pip3 install jupyter
jupyter notebook --generate-config  # ~/.jupyter/jupyter_notebook_config.py
```

**Generate passward using jupyter-console**

```
In [1]: from notebook.auth import passwd
In [2]: passwd()
```

**Jupyter config**

```
## The IP address the notebook server will listen on.
c.NotebookApp.ip = '*'   # allow ALL
  
#  The string should be of the form type:salt:hashed-password.
c.NotebookApp.password = u'sha1:96d749b4e109:17c2968d3bc899fcd41b87eb0853a42ceb48c521'
  
## The port the notebook server will listen on.
c.NotebookApp.port = 8888
 
c.NotebookApp.open_browser = False
```

### Issues

**locale.Error: unsupported locale setting**

`export LC_ALL=C`

**opengl-libs xvfb-run conflict**

https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html

    What @pemami4911 wrote on #366 (THANKS!) finally pointed me into the right direction.

    I didn't xvfb and sadly also X-dummy to work for a long time but when I followed pemami4911's hint and installed the Nvidia driver with --no-opengl-files option and CUDA with --no-opengl-libs xvfb worked right away.
    I did not have to do anything complicated, just installing drivers with --no-opengl-files and CUDA with --no-opengl-libs. Just in case I documented the necessary steps here