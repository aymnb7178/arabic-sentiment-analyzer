.PHONY: install test demo clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

demo:
	python scripts/demo.py

clean:
	rm -rf models data .pytest_cache
