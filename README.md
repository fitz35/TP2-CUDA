# installation
## python
First install anaconda (https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da), next create environnement :
'''
conda create -n myenv -f requirements.txt
'''

Activate environnement in vscode : open the command palette (Ctrl+Shift+P) and type "Python: Select Interpreter" and select the environnement.

to update the environnement :
'''
conda env update -n myenv --file requirements.txt
'''

To use jupyter notebook in vscode, activate the environnement and install the extension "Jupyter" in vscode.
## cuda
To install cuda, follow the instructions on the nvidia website (https://docs.nvidia.com/cuda/wsl-user-guide/index.html). To install the driver, you need to install the windows version of the driver (https://www.nvidia.com/Download/index.aspx?lang=en-us). To Do that get the version of your hardware in your peripheral menu. Then download the driver and install it. You havn't to install driver in wsl. Then install cuda in wsl.

# package
## numpy
https://numpy.org/doc/stable/user/index.html#user
## matplotlib
https://zestedesavoir.com/tutoriels/469/introduction-aux-graphiques-en-python-avec-matplotlib-pyplot/

## other
### execute shell command in python
'''
import subprocess

subprocess.run(["ls", "-l"])
'''

# add a custom package to pylint and vscode

In the config file of vscode (settings.json), add the path of the package to the python.linting.pylintArgs list.

'''
"python.linting.pylintArgs": [
        "--init-hook",
        "import sys; sys.path.append('${workspaceFolder}/Functions')"
    ]
'''

# C++ and openMP

# CUDA

Sur mon PC, la limite de thread est de 1024, et de block 2048.

# remarque

Pour critical et atomic, il y a un lock pour qu'une case mémoire ne soit utiliséeque par un processus à la fois. Plus il y a de cores, plus la ligne d'attente est longue. Le lock, bien que petit, cause un ralentissement. Pour le sequentiel, il n'y a pas de lock, donc pas de ralentissement.
A noter que le lock du critical est plus lent que celui du atomic donc les latences sont plus grandes.