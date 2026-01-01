
import yaml

def load_config(path='config.yaml'):
    with open(path) as f:
        try:
           return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)




