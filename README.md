# AI-Dojo

### Prerequisites

-) [Python](https://www.python.org/downloads/) 3.8 or higher <br/>
-) [pip](https://www.google.com/search?q=how+to+install+pip) <br/>
-) [numpy](https://pypi.org/project/numpy/) <br/>
-) [OpenCV](https://pypi.org/project/opencv-python/) <br/>
-) [PyTorch](https://pytorch.org/get-started/locally/) <br/>

### Training the model

Move in the "AI-Dojo" directory. </br>
Start the training script (train.py) with:

    python train.py

You can change the model by adding the "--model" parameter with the model name like:

    python train.py --model "MyCNN"

You can find the models in: "src/model.py" </br>
If you want to implement your own model do it in this script.


### Testing the model

Before you test the model you need to train one! </br>
Move in the "AI-Dojo" directory. </br>
Start the testing script (test.py) with:

    python test.py

You should see a black 512x512 pixel wide window, where you can draw numbers from 0 to 9 with your left mouse button pressed down. </br>
When pressing "a" on your keyboard you will get a prediction from your trained model in the shell (where you started the scirpt) of what the model thinks what number this is. </br>
With the "d" key you can clear the window and draw a new number. </br>
To quit the program use "q".

You can change the model by adding the "--model" parameter with the model name like:

    python test.py --model "MyCNN"