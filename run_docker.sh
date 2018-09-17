docker run -it --name irl_test -p 127.0.0.1:9090:8888 --ipc=host --mount type=bind,src="$(pwd)",target=/irl jupyter/base-notebook:latest bash
