docker_number = 1

help:
	@echo  'Build Image:'
	@echo  'make build_container USER=<username>'
	@echo  ''
	@echo  'Run Container:'
	@echo  'make start_container GPU=1,2,3 PORT=1402 USER=<username>'
	@echo  ''



build:
	sudo docker build -t $(USER)/clip-inference:$(VERSION) .

start:
	make build
	GPU=$(GPU) custom-docker_v2 run  --jupyterport $(PORT) --shm-size 128G -v /mnt/qb/bethge/shared/model_vs_human_stimuli/:/data -d $(USER)/model_vs_human_$(docker_number)
