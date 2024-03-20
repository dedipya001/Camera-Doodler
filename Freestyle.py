import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
import os
from datetime import datetime
from fpdf import FPDF

pdf = FPDF()


def open_camera(camera_view_placeholder):
    margin_left = 150
    max_x, max_y = 250 + margin_left, 50
    curr_tool = "select tool"
    start_time = True
    rad = 30
    thick = 4
    prevx, prevy = 0, 0

    # screenshot the mask
    def save_mask_as_image(mask, file_format="png"):
        os.makedirs("Freestyle", exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Freestyle/mask_{current_time}.{file_format}"
        cv2.imwrite(filename, mask)

    def get_tool(x):
        # determine tool to use
        if x < 50 + margin_left:
            return "draw"
        elif x < 100 + margin_left:
            return "erase"
        elif x < 150 + margin_left:
            return "clear"
        elif x < 200 + margin_left:
            return "save"
        else:
            return "export"

    def middle_finger_raised(y12, y9):
        return (y9 - y12) > 40

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(
        min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1
    )
    draw = mp.solutions.drawing_utils
    tools_path = "images/freestyle.png"
    tools = cv2.imread("images/freestyle.png")
    if tools is None:
        st.error(f"Error: Unable to read image file '{tools_path}'")
    else:
        tools = tools.astype("uint8")

    white_screen = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White screen mask
    mask = np.ones((480, 640), dtype=np.uint8) * 255  # Initial mask
    cap = cv2.VideoCapture(0)
    while True:
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        op = hand_landmark.process(rgb)
        if not _:
            st.write("ended")
            break
        if op.multi_hand_landmarks:
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
                x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
                if x < max_x and y < max_y and x > margin_left:
                    if start_time:
                        ctime = time.time()
                        start_time = False
                    ptime = time.time()
                    cv2.circle(frm, (x, y), rad, (0, 255, 0), 2)
                    rad -= 1
                    if (ptime - ctime) > 0.8:
                        curr_tool = get_tool(x)
                        print("your current tool set to : ", curr_tool)
                        start_time = True
                        rad = 30
                else:
                    start_time = True
                    rad = 30
                if curr_tool == "draw":
                    y12 = int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)
                    if middle_finger_raised(y12, y9):
                        cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                        prevx, prevy = x, y
                    else:
                        prevx = x
                        prevy = y
                elif curr_tool == "erase":
                    y12 = int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)
                    if middle_finger_raised(y12, y9):
                        cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                        cv2.circle(mask, (x, y), 30, 255, -1)
                elif curr_tool == "save":
                    save_mask_as_image(mask, "png")
                    curr_tool = "saved"
                elif curr_tool == "clear":
                    mask.fill(255)
                elif curr_tool == "export":
                    # take all images from Freestyle and make a pdf and save to folder Saved and delete all the images from the Freestyle Folder and make it clean
                    os.makedirs("Saved", exist_ok=True)
                    # get all the images from the Freestyle folder
                    images = []
                    for filename in os.listdir("Freestyle"):
                        if filename.endswith(".png"):
                            images.append(filename)
                    # print(images)
                    # make a pdf of all the images

                    for image in images:
                        pdf.add_page()
                        pdf.image(f"Freestyle/{image}", 0, 0, 210, 297)
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    pdf.output(f"Saved/Freestyle_{current_time}.pdf")
                    # delete all the images from the Freestyle folder
                    for image in images:
                        os.remove(f"Freestyle/{image}")
                    curr_tool = "exported"

        op = cv2.bitwise_and(frm, frm, mask=mask)
        frm[:, :, 2] = op[:, :, 2]
        frm[:, :, 1] = op[:, :, 1]
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        # Overlaying white screen on top of camera feed
        frm = cv2.addWeighted(white_screen, 0.9, frm, 0.3, 0)

        # Displaying current tool
        cv2.putText(
            frm,
            curr_tool,
            (270 + margin_left, 30),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (35, 28, 221),
            1,
        )

        # Displaying tools
        frm[:max_y, margin_left:max_x] = cv2.addWeighted(
            tools, 0.7, frm[:max_y, margin_left:max_x], 0.5, 0
        )

        camera_view_placeholder.image(frm)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break


def main():
    st.set_page_config(page_title="Freestyle")
    st.title("Freestyle Mode")
    go_back_button = st.button("Go back")

    # Check if "Go back" button is clicked
    if go_back_button:
        # Redirect to the homepage using subprocess to run another Python script
        subprocess.Popen(["streamlit", "run", "homepage.py"])
        return
    st.sidebar.markdown(
        '<div style="font-size:48px;"> Buttons </div>', unsafe_allow_html=True
    )
    camera_view_placeholder = st.empty()
    camera_button = st.sidebar.button("Open Camera")

    if camera_button:
        open_camera(camera_view_placeholder)


if __name__ == "__main__":
    main()
