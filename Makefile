IMAGE=iot-nids-benchmark

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm -v "$(CURDIR)/results:/app/results" $(IMAGE)

test:
	docker run --rm $(IMAGE) pytest -q tests/

.PHONY: build run test
