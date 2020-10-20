MKFILE_PATH:=$(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR:=$(notdir $(patsubst %/,%,$(dir $(MKFILE_PATH))))

.DEFAULT_GOAL:=jl

.PHONY: jl
jl: $(CURRENT_DIR).jl precompiled.so
	julia --project=@. --startup-file=no --history-file=no --sysimage precompiled.so -e "using Weave; weave(\"$<\", out_path = :pwd)"

precompiled.so: precompiled.jl
	julia --startup-file=no --history-file=no -e "using PackageCompiler; create_sysimage([:Plots], sysimage_path=\"$@\", precompile_execution_file=\"$<\")"

.PHONY: clean
clean:
	rm -r *.ipynb *.md *_files/

.PHONY: publish
publish: jl
	echo 'TODO'

include ../../Makefile

