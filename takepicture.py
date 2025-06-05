import cv2
import os
from datetime import datetime


def main():
    save_path = "caleb.jpg"  # You can change this if needed

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press SPACE to capture image. Press ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("Capture Face Image", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            break
        elif key == 32:  # SPACE key
            cv2.imwrite(save_path, frame)
            print(f"Image saved as '{save_path}'.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()