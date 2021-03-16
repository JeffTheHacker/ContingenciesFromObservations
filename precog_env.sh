export PRECOGROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CURDIR=$(pwd)

# Enable importing from source package.
export PYTHONPATH=$PRECOGROOT:$PYTHONPATH;
export PATH=$PRECOGROOT/scripts:$PATH

# Make CARLA visible
export PYTHONPATH=$CARLAROOT/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg:$PYTHONPATH
