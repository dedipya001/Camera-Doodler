# Camera-Doodler

This is a web application built using Streamlit, OpenCV, and MediaPipe to track hand movement and draw on the screen in real time.


## Overview

The Hand Movement Tracking Web Application allows users to interact with their webcam to track hand movements and draw on the screen. It utilizes computer vision techniques provided by OpenCV and MediaPipe to detect hand landmarks and track their movements. Users can draw on the screen by moving their hands in front of the webcam.

## Features

- Real-time hand movement tracking using the webcam.
- Drawing functionality to draw on the screen.
- Evaluating the mathematical expressions using hand movement.
- Eliminates the dependency on a writing pad to draw and teach in real-time while in the Zoom meeting or Google Meet.
- Can be integrated into any platform that uses a webcam.
- Simple and intuitive user interface built with Streamlit.
## Special Feature
- Freestyle: This is like a whiteboard on your screen in which you can show your creativity and draw anything.
- Saving functionality that allows the user to save their drawing and make a PDF to share it or use it anytime.

## Technologies Used

- **Streamlit**: Streamlit is used to create the web application interface and handle user interactions.
- **OpenCV**: OpenCV is used for image processing tasks, such as accessing the webcam feed and detecting hand landmarks.
- **MediaPipe**: MediaPipe provides hand-tracking functionality, allowing the application to accurately track hand movements.
- **Wolfram Alpha**: Wolfram Alpha's computational engine is used to perform mathematical calculations based on user input.

## Security
- **Streamlit-webrtc**: Used to create a pop-up to grant permission to use Webcamera.

## Installation

To run the application locally, follow these steps:

1. Clone the repository to your local machine:
   git clone [https://github.com/Real-Partha/Camera-Doodler.git](https://github.com/Real-Partha/Camera-Doodler.git)

2. Navigate to the project directory:
  cd hand-movement-tracking-webapp
3. Install the required dependencies using pip:
  pip install -r requirements.txt
4. Run the Streamlit application:
  streamlit run main.py
5. Open your web browser and go to [http://localhost:8501](http://localhost:8501) to access the application.

## Usage

1. Once the application is running, you will see the web interface with instructions.
2. Click the "Open camera" button to start the webcam feed.
3. Use your hand movements to draw on the screen.
4. Experiment with different hand gestures and movements to create drawings.
5. You can evaluate the expressions also.

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes.

## Acknowledgements

- Special thanks to the Streamlit, OpenCV, and MediaPipe communities for providing excellent tools and resources for building interactive web applications and computer vision projects.

---

*Created by <br>[Partha Pratim Paul](https://github.com/Real-Partha)<br>[Md Ehtesham Ansari](https://github.com/mdehteshamansari)<br>[Dedipya Goswami](https://github.com/dedipya001)<br>[Abhay Raj](https://github.com/abayraj-13)<br>[Joydeep Ghosh](https://github.com/Real-Partha)*


