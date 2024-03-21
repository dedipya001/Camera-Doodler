# from wolfram_calculator import compute_latex_expression
# import streamlit as st
# from controller import open_camera


# def cs_sidebar():
#     st.sidebar.title("Camera Doodler")
#     return None


# def cs_body():
#     camera_view_placeholder = st.empty()
#     open_camera_button = st.button("Open camera", type="primary")
#     stop_button_pressed = st.button("Stop")

#     if open_camera_button:
#         open_camera(camera_view_placeholder)

#     st.markdown("Made by Team Automatrix - HackSRM 2024")


# def display_history():
#     for expression, result in calc_history:
#         st.code(f"{expression} = {result}")


# def home_page():
#     col1, col2 = st.columns([1, 6])
#     with col1:
#         cs_sidebar()
#     with col2:
#         cs_body()
#     # with col3:
#     #     st.code("History")
#     #     display_history()


# calc_history = []


# def perform_calculation(expression):
#     # Compute the result
#     result = compute_latex_expression(expression, "9392T2-5RRYU68XKG")
#     # Append expression and result to history
#     calc_history.append((expression, result))
#     return result


# if __name__ == "__main__":
#     home_page()



import streamlit as st
from controller import open_camera
from wolfram_calculator import compute_latex_expression

# Main container styling
st.markdown(
    """
    <style>
    .stApp {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar styling
def cs_sidebar():
    st.sidebar.markdown('<div style="font-size:30px; color: #00b8ff; padding-top: 20px;"> Welcome to Camera Doodler </div>', unsafe_allow_html=True)

# Main body styling
def cs_body():
    st.markdown('<div style="font-size: 20px; color: #ffffff; padding-top: 50px;">Draw, Doodle, Create Magic with Camera Doodler!</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 16px; color: #ffffff; padding-top: 20px;">Click the button below to start unleashing your creativity:</div>', unsafe_allow_html=True)
    
    camera_view_placeholder = st.empty()  

    # Default state: access not granted
    access_granted = st.checkbox("Allow camera access", value=False)

    open_camera_button = st.button("Open camera", type="primary")
    stop_button_pressed = st.button("Stop")

    if open_camera_button:
        if access_granted:
            open_camera(camera_view_placeholder)  
        else:
            st.warning("Camera access denied.")

    st.markdown("Made by Team Automatrix - HackSRM 2024") 

    st.markdown('<p style="font-size: 16px; color: #ffffff; padding-top: 50px;">Made by Team Automatrix - HackSRM 2024</p>', unsafe_allow_html=True)

def home_page():
    cs_sidebar()
    cs_body()

calc_history = []

def perform_calculation(expression):
    # Compute the result
    result = compute_latex_expression(expression, 'YOUR_WOLFRAM_API_KEY')
    # Append expression and result