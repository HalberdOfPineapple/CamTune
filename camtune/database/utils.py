import json
import subprocess

def run_as_user(command, password: str=None, user: str=None, timeout: int=None):
    prefix = []
    if user is not None or password is not None: 
        prefix = ['sudo', '-S']
        prefix = prefix + ['-u', user] if user is not None else prefix
        command = prefix + command.split()
        command = f"echo \"{password}\" | {' '.join(command)}"

    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(timeout=timeout)

    # return ret_code, stdout, stderr, command
    return proc, stdout.strip('\n'), stderr.strip('\n'), command


def initialize_knobs(knobs_config, num) -> dict:
    if num == -1:
        with open(knobs_config, 'r') as f:
            knob_details = json.load(f)
    else:
        with open(knobs_config, 'r') as f:
            knob_tmp = json.load(f)
            i = 0

            knob_details = {}
            knob_names = list(knob_tmp.keys())
            while i < num:
                key = knob_names[i]
                knob_details[key] = knob_tmp[key]
                i = i + 1

    return knob_details
