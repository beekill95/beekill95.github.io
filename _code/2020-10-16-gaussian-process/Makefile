MKFILE_PATH:=$(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR:=$(notdir $(patsubst %/,%,$(dir $(MKFILE_PATH))))

JULIA:=julia --project=@. --startup-file=no --history-file=no

.DEFAULT_GOAL:=$(CURRENT_DIR).html
$(CURRENT_DIR).html: $(CURRENT_DIR).jl precompiled.so
	$(JULIA) --sysimage precompiled.so -e "using Weave; weave(\"$<\", out_path = :pwd)"

$(CURRENT_DIR).md: $(CURRENT_DIR).jl precompiled.so
	$(JULIA) --sysimage precompiled.so -e "using Weave; weave(\"$<\", out_path = :pwd, doctype=\"github\")"

precompiled.so: precompiled.jl
	$(JULIA) -e "using PackageCompiler; create_sysimage([:Plots], sysimage_path=\"$@\", precompile_execution_file=\"$<\")"

.PHONY: clean
clean:
	rm -r *.so *.html *.md figures

.PHONY: publish
publish: $(CURRENT_DIR).md
	echo 'TODO'

include ../../Makefile

