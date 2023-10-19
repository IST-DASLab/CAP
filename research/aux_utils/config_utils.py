import yaml


__all__ = [
    "read_config"
]


class RecursiveNamespace():
    @staticmethod
    def convert_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, d):
        for key, value in d.items():
            if type(value) == dict:
                setattr(self, key, RecursiveNamespace(value))
            elif type(value) == list:
                setattr(self, key, list(map(self.convert_entry, value)))
            else:
                setattr(self, key, value)


def read_config(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    return RecursiveNamespace(config)
