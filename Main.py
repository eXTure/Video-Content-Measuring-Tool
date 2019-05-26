from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(execution_path + "\\yolo-tiny.h5")
detector.loadModel()

custom_objects = detector.CustomObjects(person=True)

total_seconds = 0
object_visable = 0
frame_count = 0

def forFrame(frame_number, output_array, output_count):
    print("Output count for unique objects : ", output_count)
    global frame_count
    if "person" in output_count:
        frame_count += 1
        print(frame_count)

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print(count_arrays)
    global frame_count
    global object_visable
    if frame_count>15:
        object_visable += 1
    global total_seconds
    total_seconds += 1
    frame_count = 0

video_path = detector.detectCustomObjectsFromVideo(
    custom_objects=custom_objects, 
    input_file_path=(execution_path + "\\molde_short_new.mp4"),
    output_file_path=(execution_path + "\\output\\persons_detected"), 
    frames_per_second=30, 
    per_second_function=forSeconds,
    per_frame_function=forFrame, log_progress=True)

print(video_path)
print("Video duration: ", total_seconds, " seconds")
print("Object visable for ", object_visable, " seconds")