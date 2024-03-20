from wolfram_calculator import compute_latex_expression
import streamlit as st
from controller import open_camera


def cs_sidebar():
    st.sidebar.title("Camera Doodler")
    return None


def cs_body():
    camera_view_placeholder = st.empty()
    open_camera_button = st.button("Open camera", type="primary")
    stop_button_pressed = st.button("Stop")

    if open_camera_button:
        open_camera(camera_view_placeholder)

    st.markdown("Made by Team Automatrix - HackSRM 2024")


def display_history():
    for expression, result in calc_history:
        st.code(f"{expression} = {result}")


def home_page():
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        cs_sidebar()
    with col2:
        cs_body()
    with col3:
        st.code("History")
        display_history()


calc_history = []


def perform_calculation(expression):
    # Compute the result
    result = compute_latex_expression(expression, "9392T2-5RRYU68XKG")
    # Append expression and result to history
    calc_history.append((expression, result))
    return result


if __name__ == "__main__":
    home_page()
