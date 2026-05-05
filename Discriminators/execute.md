# From your local machine, rsync the project (excludes .git, venvs, data):
rsync -avz --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  -e 'SSH_AUTH_SOCK= ssh -F /dev/null -i <path/to/privkey> -oProxyCommand="ssh tunnel@login.dos.cit.tum.de -i <path/to/privkey> -W %h:%p"' \
  /home/manosgior/Documents/GitHub/Oraqle/ \
  <yourusername>@graham.dos.cit.tum.de:~/Oraqle/


SSH_AUTH_SOCK= ssh -F /dev/null -i <path/to/privkey> \
  -oProxyCommand="ssh tunnel@login.dos.cit.tum.de -i <path/to/privkey> -W %h:%p" \
  <yourusername>@graham.dos.cit.tum.de

cd ~/Oraqle
docker build -t oraqle .

docker run --gpus all -it --rm \
  -v /home/manosgior/qubit_readout_klinq/data/five_qubit_data:/data/five_qubit_data:ro \
  -v /home/sandra:/data/cnn:ro \
  -v ~/oraqle_models:/app/Discriminators/saved_models\
  oraqle bash


cd /app/Discriminators && python runners/hyper_optimize.py
