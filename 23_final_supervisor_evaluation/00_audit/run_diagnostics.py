import os
import subprocess
import json

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore')
    except Exception as e:
        return str(e)

out = {
    'nvidia_smi': run_cmd('nvidia-smi'),
    'wsl_status': run_cmd('wsl --status'),
    'wsl_list': run_cmd('wsl -l -v'),
    'python_procs': run_cmd('wmic process where "name like \'%python%\'" get processid,commandline')
}
with open(r'C:\pd-speech-crosslingual\23_final_supervisor_evaluation\00_audit\diagnostics_output.txt', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=4)
print("Diagnostics saved to diagnostics_output.txt")
