format:
	black reject/

# mypy:
# 	mypy -p tools
# 	mypy main.py
# 	mypy -p data_loader
# 	mypy -p model
# 	mypy -p trainer
# 	mypy -p metrics
# 	mypy -p optimizer

test:
	pytest tests/ --cov=reject