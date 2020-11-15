all: zip

zip:
	DIR=$(CURDIR)
	cd $(DIR)
	rm -rf projects/dlsa.zip
	zip -r projects/dlsa.zip  dlsa/ setup.py -x "**/__pycache__/*" "**/.git/**"
