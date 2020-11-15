all: zip

zip:
	GIT_VERSION := "$(shell git describe --abbrev=6 --always --tags)"
	DIR=$(CURDIR)
	cd $(DIR)
	rm -rf projects/dlsa.zip
	zip -r projects/dlsa.zip  dlsa/ setup.py README.md LICENSE  -x "**/__pycache__/*" "**/.git/**" "**/*_out"
