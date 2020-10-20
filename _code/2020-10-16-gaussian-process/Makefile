MKFILE_PATH:=$(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR:=$(notdir $(patsubst %/,%,$(dir $(MKFILE_PATH))))

include ../../Makefile

JULIA:=julia --project=$(ROOT_DIR) --startup-file=no --history-file=no

.DEFAULT_GOAL:=$(CURRENT_DIR).html
$(CURRENT_DIR).html: $(CURRENT_DIR).jl precompiled.so
	$(JULIA) --sysimage precompiled.so -e "using Weave; weave(\"$<\", out_path = :pwd)"

$(CURRENT_DIR).md: $(CURRENT_DIR).jl precompiled.so
	$(JULIA) --sysimage precompiled.so -e "using Weave; weave(\"$<\", out_path = :pwd, doctype=\"github\")"

precompiled.so: precompiled.jl
	$(JULIA) -e "using PackageCompiler; create_sysimage([:Plots], sysimage_path=\"$@\", precompile_execution_file=\"$<\")"

.PHONY: publish
publish: $(CURRENT_DIR).md
	cp $< $(POSTS_DIR)
	cp -TRv figures $(IMAGES_DIR)/$(CURRENT_DIR)

	# Change image paths in the published post.
	sed -r 's/(!\[.*\]\()figures\/([^)]+\))/\1{{ site.baseUrl }}\/images\/$(CURRENT_DIR)\/\2/g' -i "$(POSTS_DIR)/$<"

.PHONY: clean
clean:
	rm -r *.so *.html *.md figures
