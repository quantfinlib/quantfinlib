.PHONY: format lint major minor patch build publish docs test

help:
	@echo "Usage:"
	@echo "  make [format, lint, major, minor, patch, build, publish, docs, test]"

format:
	@echo "Formatting code with isort"
	poetry run isort .
	@echo "Formatting code with black"
	poetry run python -m black mlops

lint:
	@echo "Running flake8"
	poetry run flake8 mlops --per-file-ignores="__init__.py:F401" --ignore="D205" --docstring-convention="numpy" --max-line-length=120

major:
	poetry run bump2version major --allow-dirty --verbose
	git push --atomic origin main --tags

minor:
	poetry run bump2version minor --allow-dirty --verbose
	git push --atomic origin main --tags

patch:
	poetry run bump2version patch --allow-dirty --verbose
	git push --atomic origin main --tags

build:
	poetry build

publish:
	poetry publish

docs:
	@echo "Building docs"
	poetry run sphinx-build -M html docs/source docs/build

test:
	@echo "Running tests"
	poetry run pytest --cov=quantfinlib --cov-branch --cov-report=term-missing --cov-report=xml:coverage.xml -v tests
