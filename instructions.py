import streamlit as st


def instructions():
    st.write("Follow the Instructions : ")

    st.write(" #### 1. You can move around the Canvas by moving only one finger. ")

    st.markdown("""---""")

    st.write(
        " #### 2. If you want to write anything on the canvas, move your finger to the below button. CLICK IT. "
    )
    st.image("images/pen.png")
    st.write(" #### Then, raise your middle finger to start writing. ")

    st.markdown("""---""")

    st.write(
        " #### 3. If you want to erase anything on the canvas, move your (one) finger to the below button. CLICK IT. "
    )
    st.image("images/eraser.png")
    st.write(" #### Then, raise your middle finger to start erasing. ")

    st.markdown("""---""")

    st.write(
        " #### 4. If you want to clear the canvas, move your (one) finger to the below button. CLICK IT. "
    )
    st.image("images/clear.png")
    st.write(" #### One click will erase everything on the canvas. ")

    st.markdown("""---""")

    st.write(
        " #### 5. If you want to solve the equation, move your (one) finger to the below button. CLICK IT. "
    )
    st.image("images/solve.png")

    st.markdown("""---""")

    st.write(
        " #### 6. If you want to bookmark the canvas, move your (one) finger to the below button. CLICK IT. "
    )
    st.image("images/save.png")

    st.markdown("""---""")

    st.write(
        " Note: If u are solving any mathematical operations, first save it, then solve it. this will help you in understanding well."
    )
    st.write(
        " Note: Dont go out of the screen while doodling. It will not come on the screen."
    )


if __name__ == "__main__":
    instructions()
