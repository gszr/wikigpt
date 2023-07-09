venv:
	python3 -m venv venv

deps: venv
	( \
		. venv/bin/activate; \
		pip3 install -r requirements.txt; \
		deactivate; \
	)

run: deps
	( \
		. venv/bin/activate; \
		python3 wikigpt.py \
	)

.PHONY: venv deps run
