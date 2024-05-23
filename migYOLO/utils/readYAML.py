import yaml

'''Read YOLO ocnfiguration file'''
def read_config_file(file_path):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data
