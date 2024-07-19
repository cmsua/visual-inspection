# Hexaboard Visual Inspection
## University of Alabama

## Installation

This repository is designed to install from source.

```bash
# Clone the Repository
git clone --recursive https://github.com/cmsua/visual-inspection.git
cd visual-inspection

# Setup a Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Install nbstripout commit hook
nbstripout --install
```
