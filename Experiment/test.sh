CHECKPOINT=../Model/esp_train_results/2021-01/01-24-20-31-06_Left_Turn_Dataset_precog.bijection.social_convrnn.SocialConvRNN_/esp-model-668000
MODEL=../Model/esp_train_results/2021-01/01-24-20-31-06_Left_Turn_Dataset_precog.bijection.social_convrnn.SocialConvRNN_

# This is left turn with contingent planner at location 0.
python scenario_runner.py \
--enable-inference \
--enable-control \
--enable-recording \
--checkpoint_path $CHECKPOINT \
--model_path $MODEL \
--replan 4 \
--planner_type 0 \
--scenario 0 \
--location 0 \
--video_name leftturn \
--max_episodes 10

# This is overtake with contingent planner at location 2
python scenario_runner.py \
--enable-inference \
--enable-control \
--enable-recording \
--checkpoint_path $CHECKPOINT \
--model_path $MODEL \
--replan 4 \
--planner_type 0 \
--scenario 1 \
--location 2 \
--video_name overtake \
--max_episodes 10

# This right turn turn with contingent planner at location 3
python scenario_runner.py \
--enable-inference \
--enable-control \
--enable-recording \
--checkpoint_path $CHECKPOINT \
--model_path $MODEL \
--replan 4 \
--planner_type 0 \
--scenario 2 \
--location 3 \
--video_name rightturn \
--max_episodes 10

# This is overtake with overconfident planner at location 2
python scenario_runner.py \
--enable-inference \
--enable-control \
--enable-recording \
--checkpoint_path $CHECKPOINT \
--model_path $MODEL \
--replan 4 \
--planner_type 1 \
--scenario 1 \
--location 2 \
--video_name overtakeoverconfident \
--max_episodes 3

# Finally this is right turn with underconfident planner at location 0 The first 5 episodes will be run to collect data, and the 6 being the actual experiment.
python scenario_runner.py \
--enable-inference \
--enable-control \
--enable-recording \
--checkpoint_path $CHECKPOINT \
--model_path $MODEL \
--replan 4 \
--planner_type 2 \
--scenario 2 \
--location 0 \
--video_name rightturnunderconfident \
--max_episodes 6

