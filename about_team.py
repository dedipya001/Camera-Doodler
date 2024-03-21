
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
            <img src="https://media.licdn.com/dms/image/D5603AQGhTDwPLedjTg/profile-displayphoto-shrink_400_400/0/1710998788244?e=1716422400&v=beta&t=OqAV38DL-Uj_sXvvcjP2VPQeBvXpc1OyDmCNXH9NUzI" alt="Team Member 1">
                <div>Abhay Raj</div>
            </div>
            <div class="member">
                <img src="https://media.licdn.com/dms/image/D4E03AQFnu14mAmTtKQ/profile-displayphoto-shrink_400_400/0/1690468411854?e=1716422400&v=beta&t=_dCtYJyEYhIvLNmtmCtH_P3zGAENFLHh6F2oKq6x-ww" alt="Team Member 2">
                <div>Dedipya Goswami</div>
            </div>
            <div class="member">
                <img src="https://media.licdn.com/dms/image/C5603AQHp6LAyPjxXMQ/profile-displayphoto-shrink_400_400/0/1662826585542?e=1716422400&v=beta&t=LxFQL1DY2RsHG-3izL7w0f8Rdl--WtrEbhCW7_WesFg" alt="Team Member 3">
                <div>Joydeep Ghosh</div>
            </div>
            <div class="member">
                <img src="https://media.licdn.com/dms/image/D5635AQHQWSrlxCT2LQ/profile-framedphoto-shrink_800_800/0/1677694848596?e=1711605600&v=beta&t=XvubaRNNbMR3AAQMnsZmMcLeU4EorkFIcAhF3IpW998" alt="Team Member 4">
                <div>Md Ehtesham Ansari</div>
            </div>
            <div class="member">
                <img src="https://media.licdn.com/dms/image/C4D03AQGO-nnOkK-xPg/profile-displayphoto-shrink_400_400/0/1655298529958?e=1716422400&v=beta&t=TZwhsH0bIEmG_j0ziKqHaAus2wVE_5rzJ120g_UTr2I" alt="Team Member 5">
                <div>Partha Pratim Paul</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    about_project()
    about_team()
