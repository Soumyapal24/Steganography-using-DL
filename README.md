# Steganography using Deep Learning

This project demonstrates a deep learning-based approach to steganography, where a secret image is hidden within a cover image. The application uses a pre-trained deep learning model to encode and decode images.

## Project Overview

Steganography is the practice of hiding a secret message within another medium. In this project, we use a deep learning model to hide a secret image within a cover image. The model is trained to encode the secret image into the cover image and decode it back.

## Features

- Upload a cover image and a secret image.
- Encode the secret image into the cover image using a deep learning model.
- Display the encoded (stego) image.
- Calculate and display the pixel difference (loss) between the cover image and the stego image.
- Optionally, decode the secret image from the stego image (if a decoding model is available).

## Requirements

To run this project locally, you need the following dependencies:

- Python 3.x
- Streamlit
- TensorFlow
- NumPy
- Pillow
- OpenCV

You can install the required packages using the following command:

```sh
pip install -r requirements.txt
```
Additionally, you need to install libgl1-mesa-glx for OpenCV to work correctly. You can install it using the following command:

```sh
sudo apt-get install libgl1-mesa-glx
```
## Requirements How to Run

1. Clone the repository:

```sh
git clone https://github.com/yourusername/Steganography-using-DL.git
cd Steganography-using-DL
```
2. Install the required packages: 

```sh
pip install -r requirements.txt
```

3. Run the Streamlit application:

```sh
streamlit run app.py
```

4. Open your web browser and go to http://localhost:8501 to access the application.

## Usage

1. Upload a cover image using the "Upload the Cover Image" file uploader.
2. Upload a secret image using the "Upload the Secret Image" file uploader.
3. The application will preprocess the images and encode the secret image into the cover image.
4. The encoded (stego) image will be displayed.
5. The pixel difference (loss) between the cover image and the stego image will be calculated and displayed.

## Live Demo

You can access the live demo of the application at the following URL:

https://steganography-using-dl.streamlit.app/

# Contributors

- Soumya Pal: Roll No: 231043, Email: soumya.cs23@duk.ac.in
- Marcie M: Roll No: 231030, Email: marcie.cs23@duk.ac.in

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
