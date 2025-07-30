#!/bin/bash
tmux new -s sfc_train "
  trap 'echo Stopping training...' EXIT
  caffeinate -is python train_agent.py | tee training.log
"