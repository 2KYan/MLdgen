import re
import yaml
import dgm

class config:
    def __init__(self, filename, cfg_string_list):
        self.cfg_name = dgm.find_file(filename)

        with open(self.cfg_name) as yamlstream:
            self.data = yaml.load(yamlstream)

        string_list = cfg_string_list.split(";")
        for cfg_string in string_list:
            cfg_string = re.sub("\s*(\w+)\s+(.*)", r"\1: \2", cfg_string)
            cfg_string = re.sub(" y", " yes", cfg_string)
            cfg_string = re.sub(" n", " no", cfg_string)
            pair_values = yaml.load(cfg_string)
            if pair_values:
                for key in pair_values.keys():
                    self.data[key] = pair_values[key]

    def __getitem__(self, item):
        return self.get(item)

    def get(self, name):
        return self.data[name]

    def save(self, dump_dir):
        with open(dump_dir + "/config.yaml", "w") as outfile:
            yaml.dump(self.data, outfile, default_flow_style=False)
