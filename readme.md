# Contingencies From Observations

## Purposes

1. Serve as the accompanying code for ICRA 2021 paper: Contingencies from Observations.
2. A framework for running scenarios with PRECOG models in CARLA.

## Installing CARLA

This repository requires CARLA 0.9.8. Please navigate to carla.org to download the correct packages, or do the following:
```bash
# Downloads hosted binaries
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.8.tar.gz

# Unpack CARLA 0.9.8 download
tar -xvzf CARLA_0.9.8.tar.gz -C /path/to/your/desired/carla/install
```

Once downloaded, make sure that `CARLAROOT` is set to point to your copy of CARLA:
```bash
export CARLAROOT=/path/to/your/carla/install
```

`CARLAROOT` should point to the base directory, such that the output of `ls $CARLAROOT` shows the following files:
```bash
CarlaUE4     CHANGELOG   Engine  Import           LICENSE                        PythonAPI  Tools
CarlaUE4.sh  Dockerfile  HDMaps  ImportAssets.sh  Manifest_DebugFiles_Linux.txt  README     VERSION
```

## Setup

```bash
conda create -n precog python=3.6.6
conda activate precog
# make sure to source this every time after activating, and make sure $CARLAROOT is set beforehand
source precog_env.sh
pip install -r requirements.txt
```
Note that `CARLAROOT` needs to be set and `source precog_env.sh` needs to be run every time you activate the conda env in a new window/shell.

Before running any of the experiments, you need to launch the CARLA server:
```bash
cd $CARLAROOT
./CarlaUE4.sh
```

## Downloading the CARLA dataset

The training data used to train the models in the paper can be downloaded [at this link](https://drive.google.com/file/d/14-o8XZtqJnRRCPqX3gz-LJuOgBORcbXT/view?usp=sharing).

## Collecting the CARLA dataset

Alternatively, data can be generated in CARLA via the `scenario_runner.py` script:
```bash
cd Experiment
python scenario_runner.py \
--enable-collecting \
--scenario 0 \
--location 0  
```
Episode data will be stored to Experiment/Data folder.

Then run:
```bash
cd Experiment
python Utils prepare_data.py
```
This will convert the episode data objects into json file per frame, and store them in Data/JSON_output folder.

## CfO model 

The CfO model/architecture code is contained in the [precog](precog) folder, and is based on the [PRECOG repository](https://github.com/nrhine1/precog) with several key differences:

1. The architecture makes use of a CNN to process the LiDAR range map for contextual input instead of a feature map (see [precog/bijection/social_convrnn.py](precog/bijection/social_convrnn.py)). 
2. The social features also include velocity and acceleration information of the agents (see [precog/bijection/social_convrnn.py](precog/bijection/social_convrnn.py)).
3. The plotting script visualizes samples in a fixed set of coordinates with LiDAR overlayed on top (see [precog/plotting/plot.py](precog/plotting/plot.py)). 

## Running experiments with the CfO model

```bash
cd Experiment
python scenario_runner.py \
--enable-inference \
--enable-control \
--enable-recording \
--checkpoint_path [absolute path to model checkpoint] \
--model_path [absolute path to model folder] \
--replan 4 \
--planner_type 0 \
--scenario 0 \
--location 0
```

The example script [test.sh](Experiment/test.sh) will run the experiments from the paper and generate a video for each one. For reference, when using a Titan RTX GPU and Intel i9-10900k CPU each episode takes approximately 10 minutes to run, and the entire script takes several hours to run to completion.

## Training model

Organize the json files into the following structure:

```md
Custom_Dataset
---train
   ---feed_Episode_1_frame_90.json
   ...
---test
   ...
---val
   ...
```

Modify relevant precog/conf files to insert correct absolute paths.

```md
Custom_Dataset.yaml
esp_infer_config.yaml
esp_train_config.yaml
shared_gpu.yaml
sgd_optimizer.yaml # set training hyperparameters
```

Then run

```bash
export CUDA_VISIBLE_DEVICES=0;
python $PRECOGROOT/precog/esp_train.py \
dataset=Custom_Dataset \
main.eager=False \
bijection.params.A=2 \
optimizer.params.plot_before_train=True \
optimizer.params.save_before_train=True
```

## Running the MFP baseline

Install the [MFP baseline repo](https://github.com/cpacker/multiple-futures-prediction-carla), and set `MFPROOT` to point to your copy:
```bash
export MFPROOT=/your/copy/of/mfp
```

Use the `scenario_runner_mfp.py` script to run the MFP model inside of the CARLA scenarios:
```bash
# left turn
python scenario_runner_mfp.py \
--enable-inference \
--enable-control \
--enable-recording \
--replan 4 \
--scenario 0 \
--location 0 \
--mfp_control \
--mfp_checkpoint CARLA_left_turn_scenario

# right turn
python scenario_runner_mfp.py \
--enable-inference \
--enable-control \
--enable-recording \
--replan 4 \
--scenario 2 \
--location 0 \
--mfp_control \
--mfp_planning_choice highest_score_weighted \
--mfp_checkpoint CARLA_right_turn_scenario

# overtake
python scenario_runner_mfp.py \
--enable-inference \
--enable-control \
--enable-recording \
--replan 4 \
--scenario 1 \
--location 0 \
--mfp_control \
--mfp_planning_choice highest_score_weighted \
--mfp_checkpoint CARLA_overtake_scenario
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

