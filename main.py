import streamlit as st
import subprocess

def cs_sidebar():
    st.sidebar.title("Camera Doodler")
    # st.sidebar.code('History')

def main():    
    st.title("Welcome to Camera Doddler")
    st.sidebar.write("#### Nav Bar")

    page = st.sidebar.radio("Go to", ("Home", "About Team", "Instructions"))

    if page == "Home":
        from homepage import home_page  # Place the transcript below the navigation bar
        # Place the transcript below the navigation bar
        home_page()
        if st.sidebar.button("Freestyle Mode"):
            subprocess.Popen(["streamlit", "run", "Freestyle.py"])  # Execute the main4TRY.py script
    elif page == "About Team":
        from about_team import about_team
        cs_sidebar()  # Place the transcript below the navigation bar
        about_team()
    elif page == "Instructions":
        from instructions import instructions
        cs_sidebar()  # Place the transcript below the navigation bar
        instructions()

if __name__ == '__main__':
    main()
