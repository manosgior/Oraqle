# From your local machine, rsync the project (excludes .git, venvs, data):
rsync -avz --exclude='.git' --exclude='.venv' --exclude='__pycache__' -e 'SSH_AUTH_SOCK= ssh -F /dev/null -i ~/.ssh/id_rsa -oProxyCommand="ssh tunnel@login.dos.cit.tum.de -i ~/.ssh/id_rsa -W %h:%p"' /home/manosgior/Documents/GitHub/Oraqle/ manos@graham.dos.cit.tum.de:~/Oraqle/


SSH_AUTH_SOCK= ssh -F /dev/null -i <path/to/privkey> \
  -oProxyCommand="ssh tunnel@login.dos.cit.tum.de -i <path/to/privkey> -W %h:%p" \
  <yourusername>@graham.dos.cit.tum.de

cd ~/Oraqle
docker build -t oraqle .

docker run -d --device=nvidia.com/gpu=all   -v /home/manosgior/qubit_readout_klinq/data/five_qubit_data:/data/five_qubit_data:ro   -v ~/oraqle_models:/app/saved_models   -v ~/oraqle_reports:/app/optimization_reports  -v ~/oraqle_cnn_data:/data/cnn oraqle python Discriminators/runners/hyper_optimize.py


cd /app/Discriminators && python runners/hyper_optimize.py
