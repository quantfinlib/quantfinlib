@echo off

if "%1"=="format" goto format
if "%1"=="lint" goto lint

if "%1"=="major" goto major
if "%1"=="minor" goto minor
if "%1"=="patch" goto patch

if "%1"=="build" goto build
if "%1"=="publish" goto publish


echo Usage:
echo make [format, lint, patch, major, minor, patch, build, publish]
exit /b

:format
echo formatting code with isort
poetry run isort .
echo formatting code with black
poetry run python -m black mlops
exit /b

:lint
echo run flake8
poetry run flake8 mlops --per-file-ignores="__init__.py:F401" --ignore="D205" --docstring-convention="numpy" --max-line-length=120
exit /b

:major
poetry run bump2version major --allow-dirty --verbose 
git push --atomic origin main --tags
exit /b

:minor
poetry run bump2version minor --allow-dirty --verbose 
git push --atomic origin main --tags
exit /b

:patch
poetry run bump2version patch --allow-dirty --verbose 
git push --atomic origin main --tags
exit /b

:build
poetry build
exit /b

:publish
poetry publish
exit /b
