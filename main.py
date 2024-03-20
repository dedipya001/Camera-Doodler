import streamlit as st
import subprocess


def cs_sidebar():

    st.sidebar.title("Camera Doodler")
    # st.sidebar.code('History')


def main():
    st.set_page_config(page_title="Camera Doodler")
    st.title("Welcome to Camera Doodler")
    st.sidebar.write("### Menu")

    page = st.sidebar.radio("Go to", ("Home", "About Team", "Instructions"))

    if page == "Home":
        from homepage import home_page 

        
        home_page()
        if st.sidebar.button("Freestyle Mode"):
            subprocess.Popen(
                ["streamlit", "run", "freestyle.py"]
            ) 
    elif page == "About Team":
        from about_team import about_project, about_team

        cs_sidebar()  
        about_project()
        about_team()
    elif page == "Instructions":
        from instructions import instructions

        cs_sidebar()  
        instructions()


if __name__ == "__main__":
    main()
