# Makefile for Enterprise LLM Fine-Tuning Pipeline

.PHONY: setup generate clean prepare train-sft evaluate export infer pipeline run-all docker-build deploy help

help:
	@echo "Available commands:"
	@echo "  setup       - Install dependencies"
	@echo "  generate    - Run synthetic dataset batch generation"
	@echo "  clean       - Scrub PII from generated raw files"
	@echo "  prepare     - Merge raw files and split into train/val/test splits"
	@echo "  train-sft   - Trigger Supervised Fine-Tuning adapters training"
	@echo "  evaluate    - Run model evaluation (loss + perplexity)"
	@echo "  export      - Merge LoRA adapters and package model for deployment"
	@echo "  infer       - Run inference on a sample conversation"
	@echo "  run-all     - Run full sequential data pipeline and train model"
	@echo "  docker-build- Build development docker container"
	@echo "  deploy      - Deploy AWS infrastructure via CDK"

setup:
	pip install -r requirements.txt
	pip install -r requirements/training.txt
	python -m spacy download en_core_web_sm

generate:
	python main.py generate

clean:
	python main.py clean

prepare:
	python main.py prepare

train-sft:
	python main.py train-sft

evaluate:
	python main.py evaluate

export:
	python main.py export --format merged_16bit --package-sm

infer:
	python main.py infer

run-all:
	python main.py run-all

docker-build:
	docker build -t llm-finetuning -f docker/Dockerfile .

deploy:
	cd deploy && pip install -r requirements.txt && cdk deploy --all
