import yaml

def get_config():
    with open('/home/student/Desktop/efficentnet/configs/gcvit_base.yaml', 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.CLoader)
    return cfg