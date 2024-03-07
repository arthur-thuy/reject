format:
	isort reject/ tests/ --profile black
	black reject/ tests/

mypy:
	mypy reject/

test:
	pytest tests/ --cov=reject

lint:
	flake8 reject/ tests/
