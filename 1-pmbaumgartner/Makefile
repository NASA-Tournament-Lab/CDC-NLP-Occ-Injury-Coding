.PHONY: null docker-build copy-solution zip clean

null:
	@echo "You must specify a target with this makefile!"

docker-build:
	docker build -t pmbaumgartner -f ./code/dockerfile ./code

copy-solution:
	@[ "${SOLUTION_CSV_PATH}" ] || ( echo "SOLUTION_CSV_PATH is not set"; exit 1 )
	rsync -chazP $(SOLUTION_CSV_PATH) ./solution/

SOLUTION_PATH = ./solution/solution.csv
zip:
	@if [ -f $(SOLUTION_PATH) ]; then \
		zip -r ../submission-final.zip ./*; \
	else \
		echo "No Solution at $(SOLUTION_PATH)";\
	fi

clean:
	rm $(SOLUTION_PATH)

SHELL := /bin/bash