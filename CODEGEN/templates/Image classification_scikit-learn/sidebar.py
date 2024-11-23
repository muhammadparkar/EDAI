import streamlit as st
import pandas as pd

# Define possible models in a dict.
# Format of the dict: model name -> model code
MODELS = {
    "Support vectors": "sklearn.svm.SVC",
    "Random forest": "sklearn.ensemble.RandomForestClassifier",
    "Perceptron": "sklearn.linear_model.Perceptron",
    "K-nearest neighbors": "sklearn.neighbors.KNeighborsClassifier",
    "Decision tree": "sklearn.tree.DecisionTreeClassifier",
}


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {}

    with st.sidebar:
        st.write("## Model")
        model = st.selectbox("Which model?", list(MODELS.keys()))
        inputs["model_func"] = MODELS[model]

        st.write("## Input data")
        inputs["data_format"] = st.selectbox(
            "What best describes your input data?", ("Numpy arrays", "Image files", "CSV file")
        )
        if inputs["data_format"] == "Numpy arrays":
            st.write(
                """
                Expected format: `[images, labels]`
                - `images` has array shape (num samples, color channels, height, width)
                - `labels` has array shape (num samples, )
                """
            )
        elif inputs["data_format"] == "Image files":
            st.write(
                """
                Expected format: One folder per class, e.g.
                ```
                train
                +-- dogs
                |   +-- lassie.jpg
                |   +-- komissar-rex.png
                +-- cats
                |   +-- garfield.png
                |   +-- smelly-cat.png
                ```
                
                See also [this example dir](https://github.com/jrieke/traingenerator/tree/main/data/image-data)
                """
            )
        elif inputs["data_format"] == "CSV file":
            uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    inputs["csv_data"] = data
                    st.write("Preview of the uploaded CSV file:")
                    st.write(data.head())
                except Exception as e:
                    st.error(f"Error reading the CSV file: {e}")

        st.write("## Preprocessing")
        if inputs["data_format"] == "Image files":
            inputs["resize_pixels"] = st.number_input(
                "Resize images to... (required for image files)", 1, None, 28
            )
            inputs["crop_pixels"] = st.number_input(
                "Center-crop images to... (required for image files)",
                1,
                inputs["resize_pixels"],
                min(28, inputs["resize_pixels"]),
            )
        inputs["scale_mean_std"] = st.checkbox("Scale to mean 0, std 1", True)

        st.write("## Training")
        st.write("No additional parameters")

        # st.write("## Visualizations")
        # inputs["visualization_tool"] = st.selectbox(
        #     "How to log metrics?", ("Not at all", "Tensorboard", "comet.ml")
        # )
        # if inputs["visualization_tool"] == "comet.ml":
        #     "[Sign up for comet.ml](https://www.comet.ml/) :comet: "
        #     inputs["comet_api_key"] = st.text_input("Comet API key (required)")
        #     inputs["comet_project"] = st.text_input("Comet project name (optional)")
        # elif inputs["visualization_tool"] == "Tensorboard":
        #     st.markdown(
        #         "<sup>Logs are saved to timestamped dir in `./logs`. View by running: `tensorboard --logdir=./logs`</sup>",
        #         unsafe_allow_html=True,
        #     )

    return inputs


if __name__ == "__main__":
    inputs = show()
    st.write("Collected inputs:", inputs)
