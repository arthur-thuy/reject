format:
	black reject/

mypy:
	mypy reject/

test:
	pytest tests/ --cov=reject
