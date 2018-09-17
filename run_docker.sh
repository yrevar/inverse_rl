docker run -it --name irl_test -p 127.0.0.1:9090:8888 --ipc=host -e GRANT_SUDO=yes --mount type=bind,src="$(pwd)/dataset",target=/home/jovyan/inverse_rl/dataset jupyter/base-notebook-irl:latest bash
