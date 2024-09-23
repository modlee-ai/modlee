# Minimal makefile for Sphinx documentation

# Environment setup
# 	1)	Setup virtual environment. Ex: python3 -m venv venv, source venv/bin/activate
# 	2)	from root modlee dir, run: pip3 install .
# 	3)	cd docs/
# 	4)	now you can run the following Make commands

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


serve:
	python3 -m http.server 7777 -d build/html

apidoc:
	sphinx-apidoc -f -o source/modules ../src/modlee --separate

rebuild:
	make nb_rst && make md_rst && make apidoc && make clean && make html
	echo 'docs.modlee.ai' >> build/html/CNAME

md_rst:
	pandoc -s README.md -o source/README.rst
	pandoc -s tutorial.md -o source/tutorial.rst
	pandoc -s guides.md -o source/guides.rst
	pandoc -s troubleshooting.md -o source/troubleshooting.rst
	pandoc -s support.md -o source/support.rst
	pandoc -s optimizing_ml_guide.md -o source/optimizing_ml_guide.rst

nb_rst:
	jupyter nbconvert --to rst --output-dir ./source/notebooks/ ../examples/*.ipynb

docpush:
	- git push origin --delete gh-pages
	- git branch -d gh-pages
	cd .. && git subtree push --prefix docs/build/html origin gh-pages