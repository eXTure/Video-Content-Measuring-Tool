from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(execution_path + "\\yolo-tiny.h5")
detector.loadModel()

custom_objects = detector.CustomObjects(person=True)

total_seconds = 0

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    global total_seconds
    total_seconds += 1

video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=(execution_path + "\\molde.avi"),
                                output_file_path=(execution_path + "\\output\\persons_detected")
                                , frames_per_second=20, per_second_function=forSeconds, log_progress=True)      #somehow 20fps takes as long to process as 30fps
print(video_path)
print("Total seconds: ", total_seconds)