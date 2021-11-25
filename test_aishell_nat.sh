EXP_HOME=$(cd `dirname $0`; pwd)
cd $EXP_HOME

cd $EXP_HOME/fastcorrect

sudo pip install -r requirements.txt

sudo pip install --editable .
cd $EXP_HOME

SAVE_DIR=checkpoints/aishell_nat
export PYTHONPATH=/home/fastcorrect:$PYTHONPATH

mkdir -p ${SAVE_DIR}/log_aishell
edit_thre=-1.0

export CUDA_VISIBLE_DEVICES=0
nohup python -u eval_aishell.py "dev" "" ${SAVE_DIR} 0 >> ${SAVE_DIR}/log_aishell/nohup.b0t${edit_thre}test00.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup python -u eval_aishell.py "test" "" ${SAVE_DIR} 0 >> ${SAVE_DIR}/log_aishell/nohup.b0t${edit_thre}test01.log 2>&1 &
wait

