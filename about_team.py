
import streamlit as st

def about_project():
    st.markdown(
        """
        <style>
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            padding: 50px;
            background-color: #121212;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-bottom: 50px; /* Added margin bottom */
        }

        .title {
            font-size: 2.5em;
            margin-bottom: 30px;
        }

        .content {
            font-size: 1.2em;
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
        }

        .highlight {
            color: #00b8ff;
        }

        /* Styling for the team container */
        .team-container {
            background-color: #1a1a1a;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
        }

        .team-title {
            font-size: 2em;
            margin-bottom: 20px;
            color: #00b8ff; /* Highlight color */
        }

        .member {
            font-size: 1.2em;
            margin-bottom: 10px;
            color: #ffffff;
        }

        .member img {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="container">
            <div class="title">Camera Doodler</div>
            <div class="content">
                <p>Welcome to Camera Doodler - where creativity meets technology!</p>
                <p>Camera Doodler is a cutting-edge web application that allows you to unleash your creativity without even touching the screen. Simply use your finger movements to draw, sketch, and doodle in the air, and watch your creations come to life on your laptop screen.</p>
                <p>With Camera Doodler, there are no limits to your imagination. Whether you're an aspiring artist, a seasoned doodler, or just looking for a fun way to express yourself, Camera Doodler has you covered.</p>
                <p>So why wait? Dive into the world of touchless creativity with Camera Doodler today!</p>
                <p><span class="highlight">Experience the future of art - only with Camera Doodler.</span></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def about_team():
    st.markdown(
        """
        <div class="team-container">
            <div class="team-title">Meet the Team</div>
            <div class="member">
                <img src="https://via.placeholder.com/100" alt="Team Member 1">
                <div>Abhay Raj</div>
            </div>
            <div class="member">
                <img src="https://via.placeholder.com/100" alt="Team Member 2">
                <div>Dedipya Goswami</div>
            </div>
            <div class="member">
                <img src="https://via.placeholder.com/100" alt="Team Member 3">
                <div>Joydeep Ghosh</div>
            </div>
            <div class="member">
                <img src="https://via.placeholder.com/100" alt="Team Member 4">
                <div>Md Ehtesham Ansari</div>
            </div>
            <div class="member">
                <img src="https://via.placeholder.com/100" alt="Team Member 5">
                <div>Partha Pratim Paul</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    about_project()
    about_team()

