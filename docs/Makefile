# Minimal makefile for Sphinx documentation
#

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
	make nb_rst && make apidoc && make clean && make html

md_rst:
	pandoc -s README.md -o source/README.rst

nb_rst:
	jupyter nbconvert --to rst --output-dir ./source/notebooks/ ../examples/*.ipynb
