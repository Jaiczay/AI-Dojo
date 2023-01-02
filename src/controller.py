import os

import cv2
import numpy as np
from torchvision.transforms import ToTensor

from src.model import *


class Controller:

    def __init__(self, config):
        # extract first the parameters to create the designers
        # input
        self.input_size = (config["input"]["W"], config["input"]["H"])
        diagonal = sum([x ** 2 for x in self.input_size]) ** 0.5
        self.thickness = int(config["input"]["thickness"] * diagonal) + 1
        # output
        self.output_size = (config["output"]["W"], config["output"]["H"])
        # line
        self.range_value = self.to_list(config["line"]["range_value"])
        self.fading = config["line"]["fading"]
        self.volume = config["process"]["volume"]
        self.selection = config["process"]["selection"].upper()
        self.model = eval(config["model"] + "()")
        self.model_name = config["model"]

        # to finish, set the dynamic part
        self.last_pos = None   # to draw lines between last and current mouse position
        self.left_button_down = False  # to draw only when the mouse is pressed
        # intialize the OpenCV windows with the ouput one if asked
        cv2.namedWindow('Writing Frame', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Writing Frame', self.mouse_event) # catch mouse events

        self.img = self.set_new_image()

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # activate draw
            self.left_button_down = True
            self.draw(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.left_button_down:  # draw
            self.draw(x, y)
        elif event == cv2.EVENT_LBUTTONUP:  # deactivate draw
            self.last_pos = None
            self.left_button_down = False

    def to_list(self, x):
        # convert to list if it is not yet
        x = x if isinstance(x, list) else [x]
        return x

    def set_new_image(self):
        shape = list(self.input_size)[::-1]
        return np.zeros(shape)

    def draw(self, x, y):
        # avoid last pos non existing
        if self.last_pos is None:
            self.last_pos = (x, y)
        # draw a line between the last mouse position and its current
        self.img = cv2.line(self.img, self.last_pos, (x, y), 255, self.thickness)
        self.last_pos = (x, y)  # update the last pos

    def run(self):
        device = torch.device("cpu")
        self.model = self.model.to(device)
        self.model.eval()
        self.model.load_state_dict(torch.load(os.path.abspath("weights/" + self.model_name + "_best_weights.pt"))["model_state_dict"])
        transform = ToTensor()
        # draw classes until the volume is not reached
        while True:
            # display the frames of the drawing part and ouput part if asked
            cv2.imshow('Writing Frame', self.img)

            key = cv2.waitKey(1000//30) & 0xff # 30 fps and catch key pressed
            if key == ord('q') or key == 27: # Q key of Esc to quit the program
                break
            elif key == ord('a'):  # Enter key to save the draw and begin the next one
                inputs = transform(cv2.resize(self.img, self.output_size)) / 255
                inputs = torch.unsqueeze(inputs, 0).to(torch.float32)
                outputs = self.model(inputs.to(device))
                print(torch.argmax(outputs, 1).item())
            elif key == ord('d'):
                self.img = self.set_new_image()

        cv2.destroyAllWindows()  # preferable
