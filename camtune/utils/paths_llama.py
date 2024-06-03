import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_DIR = os.path.join(BASE_DIR, 'llamatune')
LLAMA_KNOB_DIR = os.path.join(LLAMA_DIR, 'spaces', 'definitions')
LLAMA_SPACE_DIR = os.path.join(LLAMA_DIR, 'spaces')
SMAC_OUTPUT_DIR = os.path.join(LLAMA_DIR, 'smac_output')

def set_smac_output_dir(expr_name: str, benchmark_name: str):
    global SMAC_OUTPUT_DIR
    SMAC_OUTPUT_DIR = os.path.join(SMAC_OUTPUT_DIR, benchmark_name, expr_name)
    os.makedirs(SMAC_OUTPUT_DIR, exist_ok=True)

def get_smac_output_dir() -> str:
    return SMAC_OUTPUT_DIR