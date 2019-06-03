from imageai.Detection import VideoObjectDetection
import os, logging

logging.basicConfig(level=logging.DEBUG, 
                filename='app.log', 
                filemode='w', 
                format='%(name)s - %(levelname)s - %(message)s')

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(execution_path + "\\yolo-tiny.h5")
detector.loadModel()

custom_objects = detector.CustomObjects(person=True)

total_seconds = 0
object_visible = 0
frame_count = 0

def forFrame(frame_number, output_array, output_count):
    global frame_count
    if "person" in output_count:
        frame_count += 1
        logging.info("Frames with visible objects: %s", frame_count)
    else:
        logging.info("No objects visible")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    logging.info("SECOND : %s", second_number)
    logging.info(count_arrays)
    global total_seconds
    total_seconds += 1

def forFull(output_arrays, count_arrays, average_output_count):
    object_visible = frame_count/30
    logging.info("Video duration: %s seconds", total_seconds)
    logging.info("Object visible for %s seconds", object_visible)
    f = open('result.txt','w')
    write_msg = ["Video duration: ", str(total_seconds), " seconds", 
                "\nObject visable for ", str(object_visible), " seconds"]
    f.writelines(write_msg)
    f.close()

def main():
    detector.detectCustomObjectsFromVideo(
        custom_objects=custom_objects, 
        input_file_path=(execution_path + "\\molde_short_new.mp4"),
        output_file_path=(execution_path + "\\output\\objects_detected"), 
        frames_per_second=30, 
        per_second_function=forSeconds,
        per_frame_function=forFrame,
        video_complete_function=forFull, log_progress=True)

if __name__ == "__main__":
    main()