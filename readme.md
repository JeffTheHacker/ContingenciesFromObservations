# Contingencies From Observations

# Purposes

1. Serve as the accompanying code for ICRA 2021 paper: Contingencies from Observations.
2. A framework for running scenarios with Precog models in Carla.

## Setup

```bash
conda create -n precog python=3.6.6
conda activate precog
source precog_env.sh # make sure to run this every time
pip install -r requirements.txt
```

## Installing Carla

This repository requires Carla 0.9.8. Please navigate to carla.org to install carla in the same conda environment.

## Running experiments with Precog model

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

## Collecting data

First collect data in Carla.

```bash
cd Experiment
python scenario_runner.py \
--enable-collecting \
--scenario 0 \
--location 0  
```

Episode data will be stored to Experiment/Data folder.
Then run

```bash
cd Experiment
python Utils prepare_data.py
```

This will convert the episode data objects into json file per frame, and store them in Data/JSON_output folder.

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


## License
[MIT](https://choosealicense.com/licenses/mit/)
