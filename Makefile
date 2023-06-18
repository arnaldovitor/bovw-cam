test:
	pytest .


lint:
	@echo
	ruff src
	@echo
	blue --check --diff --color src
	@echo
	mypy src
	@echo
	pip-audit


format:
	ruff --silent --exit-zero --fix src
	blue src