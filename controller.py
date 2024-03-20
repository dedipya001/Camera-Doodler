import numpy as np
import cv2
import mediapipe as mp
import time
from wolfram_calculator import compute_latex_expression
import streamlit as st
from segment_predict import segment_and_predict
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("new_model_whitebg.h5")


# @st.cache(
#     suppress_st_warning=True,
#     hash_funcs={st.delta_generator.DeltaGenerator: lambda _: None},
# )
def open_camera(camera_view_placeholder):

    margin_left = 150
    max_x, max_y = 250 + margin_left, 50
    curr_tool = "select tool"
    start_time = True
    rad = 30
    thick = 11
    prevx, prevy = 0, 0

    # screenshot the mask
    def save_mask_as_image(mask, file_format="png"):
        filename = f"mask_capture.{file_format}"
        cv2.imwrite(filename, mask, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    def get_tool(x):
        # determine tool to use
        if x < 50 + margin_left:
            return "draw"
        elif x < 100 + margin_left:
            return "erase"
        elif x < 150 + margin_left:
            return "clear"
        elif x < 200 + margin_left:
            return "solve"
        else:
            return "save"

    # y corresponds to the landmarks of the middle finger's landmark
    def middle_finger_raised(y12, y9):
        return (y9 - y12) > 40

    def display_numbers(list_numbers):
        int_to_str = {
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            "10": "+",
            "11": "-",
            "12": "*",
            "13": "/",
        }
        output = ""
        for key in list_numbers:
            output += int_to_str[str(key)]
        return output

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(
        min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1
    )
    draw = mp.solutions.drawing_utils
    tools_path = "images/newtools.png"
    tools = cv2.imread("images/newtools.png")
    # tools = tools.astype('uint8')
    if tools is None:
        st.error(f"Error: Unable to read image file '{tools_path}'")
    # You can choose to use a default image or handle the error in another way
    else:
        tools = tools.astype("uint8")

    mask = np.ones((480, 640)) * 255
    mask = mask.astype("uint8")

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
                    cv2.circle(frm, (x, y), rad, (0, 0, 0), 2)
                    rad -= 1
                    if (ptime - ctime) > 0.8:
                        curr_tool = get_tool(x)
                        print("you are currently using : ", curr_tool)
                        start_time = True
                        rad = 30
                else:
                    start_time = True
                    rad = 30
                if curr_tool == "draw":
                    y12 = int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)
                    if middle_finger_raised(y12, y9):
                        cv2.line(mask, (prevx, prevy), (x, y), (0, 0, 0, 0), thick)
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
                elif curr_tool == "solve":
                    save_mask_as_image(mask, "png")
                    list_numbers = segment_and_predict()
                    output = display_numbers(list_numbers)
                    answer = compute_latex_expression(output, "QY6LX3-5UPVEGR9Y9")
                    st.sidebar.latex(f"{output} \quad ={answer}")
                    curr_tool = "solved"
                elif curr_tool == "save":
                    save_mask_as_image(mask, "png")
                    list_numbers = segment_and_predict()
                    output = display_numbers(list_numbers)
                    st.sidebar.latex(output)
                    curr_tool = "saved"
                elif curr_tool == "clear":
                    mask.fill(255)

        op = cv2.bitwise_and(frm, frm, mask=mask)
        frm[:, :, 2] = op[:, :, 2]
        frm[:, :, 1] = op[:, :, 1]
        frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
        # tools = cv2.cvtColor(tools, cv2.COLOR_RGB2BGR)
        frm[:max_y, margin_left:max_x] = cv2.addWeighted(
            tools, 0.7, frm[:max_y, margin_left:max_x], 0.3, 0
        )
        cv2.putText(
            frm,
            curr_tool,
            (270 + margin_left, 30),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 0, 0),
            1,
        )
        camera_view_placeholder.image(frm)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
