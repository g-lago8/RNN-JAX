# Top-level Makefile
# Convenience targets for building the package

.PHONY: build
build:
	@echo "Calling scripts/build_rnn_jax.sh"
	@./scripts/build_rnn_jax.sh
