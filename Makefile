# Makefile for Enterprise LLM Fine-Tuning Pipeline

.PHONY: setup generate clean prepare train-sft pipeline run-all test Docker-build Help

help:
	@echo "Available commands:"
	@echo "  setup       - Install dependencies"
	@echo "  generate    - Run synthetic dataset batch generation"
	@echo "  clean       - Scrub PII from generated raw files"
	@echo "  prepare     - Merge raw files and split into train/val/test splits"
	@echo "  train-sft   - Trigger Supervised Fine-Tuning adapters training"
	@echo "  run-all     - Run full sequential data pipeline and train model"
	@echo "  docker-build- Build development docker container"

setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

generate:
	python main.py generate

clean:
	python main.py clean

prepare:
	python main.py prepare

train-sft:
	python main.py train-sft

run-all:
	python main.py run-all

docker-build:
	docker build -t llm-finetuning -f docker/Dockerfile .
