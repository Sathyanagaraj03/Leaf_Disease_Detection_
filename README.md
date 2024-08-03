# Leaf Disease Detection

This project leverages deep learning to automatically identify and classify diseases in plant leaves from images. The goal is to support agricultural practices by providing timely and accurate disease detection.

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Impact](#impact)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Leaf Disease Detection project uses a convolutional neural network (CNN) model to classify images of plant leaves into different disease categories. This innovative approach aims to assist farmers and agricultural experts in monitoring plant health and taking preventive measures against diseases.

## Objectives
- **Automate Disease Detection**: Build a system that accurately classifies leaf diseases using deep learning.
- **Enhance Crop Management**: Provide a tool for farmers to monitor plant health effectively.
- **Reduce Chemical Usage**: Minimize pesticide use through early disease detection.

## Impact
This project showcases the potential of AI to revolutionize agriculture, improve crop yields, and reduce environmental impacts by enabling more precise and sustainable farming practices.

## Installation
To run the Leaf Disease Detection application, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/leaf-disease-detection.git
    cd leaf-disease-detection
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To use the Leaf Disease Detection application:

1. **Run the Streamlit application**:
    ```sh
    streamlit run app.py
    ```

2. **Upload an image**: Use the file uploader to upload an image of a plant leaf.

3. **View Results**: The application will display the uploaded image along with the predicted disease class and confidence level.

## Model Training
If you want to train the model from scratch:

1. **Prepare the dataset**: Organize your dataset into subdirectories for each class (e.g., `Potato___healthy`, `Potato___Late_blight`, `Potato___Early_blight`).

2. **Modify the training script**: Update the training script with the correct paths to your dataset.

3. **Run the training script**:
    ```sh
    python train.py
    ```

4. **Save the model**: The trained model will be saved as a `.h5` or `.keras` file.

## Contributing
Contributions are welcome! Please follow these steps:

1. **Fork the repository**.
2. **Create a new branch**:
    ```sh
    git checkout -b feature-branch
    ```
3. **Make your changes**.
4. **Commit your changes**:
    ```sh
    git commit -m 'Add some feature'
    ```
5. **Push to the branch**:
    ```sh
    git push origin feature-branch
    ```
6. **Open a pull request**.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions or suggestions, feel free to open an issue or contact us at [your-email@example.com](mailto:your-email@example.com).

---

**Disclaimer**: This project is for educational purposes and should not be used as a substitute for professional agricultural advice.
