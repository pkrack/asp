.venv:
	python3 -m venv --upgrade-deps --prompt asp .venv
	.venv/bin/python -m pip install --upgrade pip setuptools build

install: .venv
	.venv/bin/python -m pip install -e .

clean:
	rm -rf asp.egg-info
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: install clean
