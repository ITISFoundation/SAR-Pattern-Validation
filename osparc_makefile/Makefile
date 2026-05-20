WORKSPACE_DIR  := /home/jovyan/work/workspace
REPO_DIR       := $(WORKSPACE_DIR)/SAR-Pattern-Validation
GIT_LFS_VERSION := 3.6.1
GIT_LFS_BIN    := $(HOME)/.local/bin/git-lfs

.PHONY: help
help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  setup            Run all setup steps: git-pull, install-git-lfs, lfs-pull, copy-notebook"
	@echo "  git-pull         Pull the latest code in the SAR-Pattern-Validation repository"
	@echo "  install-git-lfs  Install git-lfs via wget into ~/.local/bin (skips if already on PATH)"
	@echo "  lfs-pull         Run 'git lfs pull' inside the repository"
	@echo "  copy-notebook    Copy voila.ipynb from the repo to the workspace directory"
	@echo "  clean            Delete test run artifacts: images/, system_state/, uploaded_data/"
	@echo ""

.DEFAULT_GOAL := help

# Full setup: pull latest code, install git-lfs if needed, pull LFS files, copy notebook
.PHONY: setup
setup: git-pull install-git-lfs lfs-pull copy-notebook
	@echo "✅ Setup complete!"

# Pull the latest code from the remote
.PHONY: git-pull
git-pull:
	@echo "🔄 Pulling latest code in $(REPO_DIR)..."
	@cd $(REPO_DIR) && git pull
	@echo "✅ Repository up to date."

# Install git-lfs via wget if not already on PATH
.PHONY: install-git-lfs
install-git-lfs:
	@if command -v git-lfs > /dev/null 2>&1; then \
		echo "ℹ️  git-lfs already installed, skipping."; \
	else \
		echo "📦 Installing git-lfs $(GIT_LFS_VERSION) via wget..."; \
		wget -qO- "https://github.com/git-lfs/git-lfs/releases/download/v$(GIT_LFS_VERSION)/git-lfs-linux-amd64-v$(GIT_LFS_VERSION).tar.gz" \
			| tar -xz -C /tmp; \
		mkdir -p $(HOME)/.local/bin; \
		cp /tmp/git-lfs-$(GIT_LFS_VERSION)/git-lfs $(GIT_LFS_BIN); \
		export PATH="$(HOME)/.local/bin:$$PATH"; \
		$(GIT_LFS_BIN) install; \
		echo "✅ git-lfs installed."; \
	fi

# Pull LFS files inside the cloned repo
.PHONY: lfs-pull
lfs-pull:
	@echo "📂 Running git lfs pull in $(REPO_DIR)..."
	@export PATH="$(HOME)/.local/bin:$$PATH"; \
	cd $(REPO_DIR) && git lfs pull

# Copy voila.ipynb to the workspace directory (parent of the repo)
.PHONY: copy-notebook
copy-notebook:
	@echo "📋 Copying voila.ipynb to $(WORKSPACE_DIR)..."
	cp $(REPO_DIR)/notebooks/voila.ipynb $(WORKSPACE_DIR)/voila.ipynb
	@echo "✅ Notebook copied."

# Remove files created by a test run
.PHONY: clean
clean:
	@echo "🗑️  Cleaning test run artifacts from $(WORKSPACE_DIR)..."
	rm -rf $(WORKSPACE_DIR)/images
	rm -rf $(WORKSPACE_DIR)/report
	rm -rf $(WORKSPACE_DIR)/system_state
	rm -rf $(WORKSPACE_DIR)/uploaded_data
	@echo "✅ Clean complete!"
