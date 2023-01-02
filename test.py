import argparse
import json
import os

from src.controller import Controller


def get_json(file_name):
    # open en return the content of JSON file as a dictionary
    with open(os.path.abspath(file_name)) as json_file:
        json_data = json.load(json_file)
    return json_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="src/config.json", help="path to the config file")
    parser.add_argument("--model", default="MyNN",
                        help="Name of the model class. Default: \"MyNN\" or \"MyCNN\". "
                                                        "Classes found in src/model.py you can implement your own model there.")
    args = parser.parse_args()
    config = get_json(args.config)
    config["model"] = args.model
    controller = Controller(config)
    controller.run()
