import cv2
import os


'''
Saves video to /output folder.

version 10.20.19
'''
def save():
    cam = cv2.VideoCapture(1)

    
    path, dirs, files = next(os.walk("output"))
    file_count = len(files)
    print(file_count)

    # Set frame width and height 
    frame_width = int(cam.get(3)/2)
    frame_height = int(cam.get(4))

    # Define the codec and filename.
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  # cv2.VideoWriter_fourcc() does not exist
    out = cv2.VideoWriter("output/output{}.avi".format(file_count+1), fourcc, 24, (frame_width, frame_height))
    while (cam.isOpened()):
        ret_val, img = cam.read()
    
        # Edit video
        h,w = img.shape[:2]
        cropped = img[:, :int(w/2)]
        final_image = cv2.flip(cropped, -1)
        
        # Display Image
        out.write(final_image)
        
        cv2.imshow("Frame", final_image)
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cam.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    save()
