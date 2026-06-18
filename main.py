import streamlit as st
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import os

import streamlit as st


st.write("Current Directory:", os.getcwd())
st.write("Files:", os.listdir("."))


st.write("Current Directory:", os.getcwd())
st.write("Files:", os.listdir("."))

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="🌿",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
         "trained_plant_disease_model.keras" 
         )

import os

st.write("Current files:", os.listdir("."))

model = load_model()


# --------------------------------------------------
# DISEASE INFORMATION
# --------------------------------------------------

disease_info = {

    "Tomato___Late_blight": {
        "Symptoms": "Large brown-black lesions on leaves and stems.",
        "Treatment": "Remove infected leaves and apply fungicide."
    },

    "Tomato___Early_blight": {
        "Symptoms": "Dark circular spots with concentric rings.",
        "Treatment": "Use crop rotation and fungicide."
    },

    "Potato___Late_blight": {
        "Symptoms": "Water-soaked spots that become dark brown.",
        "Treatment": "Use certified seeds and fungicide."
    },

    "Potato___Early_blight": {
        "Symptoms": "Dark spots with target-like rings.",
        "Treatment": "Apply fungicide and remove infected leaves."
    }

}


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_name = [

'Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',

'Blueberry___healthy',

'Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy',

'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',

'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy',

'Orange___Haunglongbing_(Citrus_greening)',

'Peach___Bacterial_spot',
'Peach___healthy',

'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',

'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',

'Raspberry___healthy',

'Soybean___healthy',

'Squash___Powdery_mildew',

'Strawberry___Leaf_scorch',
'Strawberry___healthy',

'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus',
'Tomato___healthy'

]


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def model_prediction(test_image):

    image = tf.keras.preprocessing.image.load_img(
        test_image,
        target_size=(128,128)
    )


    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    input_arr = np.array([input_arr])


    predictions = model.predict(input_arr)


    result_index = np.argmax(predictions)


    confidence = float(
        np.max(predictions)*100
    )


    top3 = np.argsort(
        predictions[0]
    )[-3:][::-1]


    return result_index, confidence, top3, predictions[0]



# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🌿 Dashboard")


app_mode = st.sidebar.selectbox(

    "Select Page",

    [
        "Home",
        "About",
        "Disease Recognition",
        "Model Performance"
    ]

)



# --------------------------------------------------
# HOME
# --------------------------------------------------

if app_mode == "Home":


    st.title(
        "🌿 Plant Disease Recognition System"
    )


    try:

        st.image(
            "home_page.jpeg",
            use_container_width=True
        )

    except:

        pass


    st.markdown("""

Welcome to Plant Disease Recognition System.

This project uses Deep Learning CNN model
with TensorFlow to identify plant diseases.

### Features

- Upload Leaf Image
- Disease Prediction
- Confidence Score
- Top 3 Predictions
- Model Performance


### Supported Classes

38 Plant Disease Categories

""")


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

elif app_mode == "About":


    st.title("📘 About Project")


    st.markdown("""

### Dataset

- Training Images: 70,295
- Validation Images: 17,572
- Test Images: 33
- Classes: 38


### Technology

- Python
- TensorFlow
- Streamlit
- NumPy
- Matplotlib


### Objective

Identify plant diseases using Artificial Intelligence.

""")


# --------------------------------------------------
# DISEASE RECOGNITION
# --------------------------------------------------

elif app_mode == "Disease Recognition":


    st.title("🔬 Disease Recognition")


    test_image = st.file_uploader(

        "Upload Plant Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )


    if test_image:


        st.image(

            test_image,

            caption="Uploaded Image",

            width=400

        )


        if st.button("Predict Disease"):


            with st.spinner(
                "Analyzing Image..."
            ):


                result_index, confidence, top3, pred_array = model_prediction(test_image)


                disease = class_name[result_index]



            st.success(

                f"Predicted Disease: {disease}"

            )


            st.info(

                f"Confidence: {confidence:.2f}%"

            )



            st.subheader(
                "Top 3 Predictions"
            )


            for idx in top3:


                st.write(

                    f"{class_name[idx]} : {pred_array[idx]*100:.2f}%"

                )



            if disease in disease_info:


                st.subheader(
                    "Disease Information"
                )


                st.write(

                    "**Symptoms:**",

                    disease_info[disease]["Symptoms"]

                )


                st.write(

                    "**Treatment:**",

                    disease_info[disease]["Treatment"]

                )


            st.balloons()



# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif app_mode == "Model Performance":


    st.title(
        "📈 Model Performance"
    )


    try:


        with open(
            "training_hist.json",
            "r"
        ) as f:

            history = json.load(f)



        st.metric(

            "Training Accuracy",

            f"{history['accuracy'][-1]*100:.2f}%"

        )


        st.metric(

            "Validation Accuracy",

            f"{history['val_accuracy'][-1]*100:.2f}%"

        )



        epochs = range(

            1,

            len(history["accuracy"])+1

        )



        fig1, ax1 = plt.subplots()


        ax1.plot(

            epochs,

            history["accuracy"],

            label="Training"

        )


        ax1.plot(

            epochs,

            history["val_accuracy"],

            label="Validation"

        )


        ax1.legend()


        st.pyplot(fig1)



        fig2, ax2 = plt.subplots()


        ax2.plot(

            epochs,

            history["loss"],

            label="Training Loss"

        )


        ax2.plot(

            epochs,

            history["val_loss"],

            label="Validation Loss"

        )


        ax2.legend()


        st.pyplot(fig2)



    except Exception as e:


        st.error(

            f"Could not load training history: {e}"

        )
