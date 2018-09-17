tmux start-server
tmux new-session -d -s do_console01 -n htop
tmux new-window -tdo_console01:1 -n jupyter
tmux new-window -tdo_console01:2 -n gpu
tmux new-window -tdo_console01:3 -n sh


tmux send-keys -tdo_console01:0 'htop' C-m
tmux send-keys -tdo_console01:1 'source .env/bin/activate' Enter 'jupyter notebook --ip=0.0.0.0 --port=7000' C-m
tmux send-keys -tdo_console01:2 'nvidia-smi -l 2' C-m
tmux send-keys -tdo_console01:3 'source .env/bin/activate' C-m

tmux select-window -tdo_console01:0
tmux attach-session -d -tdo_console01

