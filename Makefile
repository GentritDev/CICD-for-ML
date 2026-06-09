install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo "\n## Confusion Matrix Plot" >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git config --global pull.rebase true
	git pull origin update

hf-login:
	git fetch origin update
	git reset --hard origin/update
	pip install -U "huggingface_hub[cli]"
	hf auth login --token $(HF)

push-hub:
	hf upload Genti123/Customer-Support-Classifier ./App --repo-type=space
	hf upload Genti123/Customer-Support-Classifier ./Model /Model --repo-type=space

deploy: hf-login push-hub
