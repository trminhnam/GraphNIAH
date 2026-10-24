ANTLR_VERSION ?= 4.9.3
export ANTLR4_TOOLS_ANTLR_VERSION = $(ANTLR_VERSION)

.PHONY: test compile

test:
	PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider

# Optional reference parsers; not needed by the bulk dataset v1 workflow.
compile:
	@for grammar in src/reference_parser/antlr4_grammar/*/*.g4; do \
		antlr4 -Dlanguage=Python3 -visitor "$$grammar"; \
		dir=$$(dirname "$$grammar"); \
		touch "$$dir/__init__.py"; \
	done
	@touch src/reference_parser/antlr4_grammar/__init__.py
