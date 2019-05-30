# Kaunas Coding School project

Aim of this project is to create Python program to measure how long a logo is visible in a given video.

### Important!
**Python version 3.6 is needed. (Tensorflow only supports python up to 3.6)**

**How to run**

1. Install the following libraries:
	
Tensorflow 
`pip install tensorflow` or ```pip install tensorflow-gpu``` for the gpu powered version

Numpy
`pip install numpy`
	
SciPy
`pip install scipy`
	
OpenCV
`pip install opencv-python`
	
Matplotlib
`pip install matplotlib`
	
h5py
`pip install h5py`
	
Keras
`pip install keras`
	
Pillow
`pip install pillow`
	
ImageAI
`pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl`

2. If you have your own model for detecting objects, add the model file to the same folder as the project and change the following line:
`detector.setModelPath(execution_path + "\\your_model.h5")`

3. Add the video, that you want to measure, to the same folder as the project and then edit the following line:
`input_file_path=(execution_path + "\\your_video.mp4")`

4. Run Main.py. After the process is finished you will find result in Result.txt and you can check the video, with marked objects, in the output folder.

**Resources:**
https://medium.com/@guymodscientist/image-prediction-with-10-lines-of-code-3266f4039c7a
https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html
https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md
https://github.com/OlafenwaMoses/ImageAI
https://medium.com/coinmonks/detecting-custom-objects-in-images-video-using-yolo-with-darkflow-1ff119fa002f
