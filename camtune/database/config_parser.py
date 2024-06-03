from typing import Tuple, Dict


class ConfigParser:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.knobs_from_cnf = {}

        with open(config_file, 'r') as file:
            for line in file:
                line = line.strip()
                if self._is_ignored_line(line):
                    continue

                try:
                    key, _, value = line.split()
                    self.knobs_from_cnf[key] = value
                except:
                    continue

    def _is_ignored_line(self, line: str) -> bool:
        return (line.startswith(('skip-external-locking', '[', '#')) or not line)
    
    def replace(self, tmp_file_path: str = '/tmp/tmp.cnf') -> str:
        """
        Replace the configuration file with the new settings (from self.knobs_from_cnf)
        """
        updated_keys = set()
        with open(self.config_file, 'r') as original, open(tmp_file_path, 'w') as modified:
            # Write the configuration that has been covered by existing settings
            for line in original:
                parts = line.strip().split()
                if parts and parts[0] in self.knobs_from_cnf:
                    parts[2] = self.knobs_from_cnf[parts[0]]
                    modified.write(f"{parts[0]}\t\t{parts[1]} {parts[2]}\n")
                    updated_keys.add(parts[0])
                else:
                    modified.write(line)

            # Write the configuration that has not been covered by existing settings
            for key, value in self.knobs_from_cnf.items():
                if key not in updated_keys:
                    modified.write(f"{key}\t\t= {value}\n")
        return tmp_file_path

    def set(self, key: str, value: str):
        if isinstance(value, str):
            self.knobs_from_cnf[key] = f"'{value}'"
        elif isinstance(value, float):
            self.knobs_from_cnf[key] = f"{value:.8f}"
        else:
            self.knobs_from_cnf[key] = value
    
    def write_to_tmp_file(self, knobs_to_apply: Dict[str, str], tmp_file_path: str = None) -> Tuple[list, str]:
        for key, value in knobs_to_apply.items():
            self.set(key, value)
        new_config_path = self.replace(tmp_file_path=tmp_file_path)
        return new_config_path
