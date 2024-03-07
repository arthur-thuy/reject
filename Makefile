format:
	isort reject/ --profile black
	black reject/

mypy:
	mypy reject/

test:
	pytest tests/ --cov=reject
