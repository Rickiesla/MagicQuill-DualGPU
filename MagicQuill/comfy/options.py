import os

args_parsing = False

def enable_args_parsing(enable=True):
    global args_parsing
    args_parsing = enable

# Auto-enable args parsing and inject DirectML if USE_DIRECTML=1 or AMD_GPU=1
if os.environ.get("USE_DIRECTML", os.environ.get("AMD_GPU", "0")) == "1":
    enable_args_parsing(True)
