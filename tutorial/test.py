import json
import argparse
from controller import Controller


def get_json(file_name):
    # open en return the content of JSON file as a dictionary
    with open(file_name) as json_file:
        json_data = json.load(json_file)
    return json_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        default="config.json",
                        help="path to the config file")
    args = parser.parse_args()
    config = get_json(args.config)
    controller = Controller(config)
    controller.run()
