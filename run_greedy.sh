#python react_batch_demo.py \
#--custom_cfg configs/react_sft.yaml \
#--qaf ../MARIO_EVAL/data/math_testset_annotation.json


# use step beam without value func
python solver_demo.py \
--custom_cfg configs/sbs_greedy.yaml \
--qaf ../MARIO_EVAL/data/math_testset_annotation.json
