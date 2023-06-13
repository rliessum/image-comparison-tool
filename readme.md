# Image Comparison and Processing Tool for AI Training

The Image Comparison and Processing Tool is a JavaScript-based web browser application designed to streamline the image comparison and processing process, particularly for preparing datasets for artificial intelligence (AI) training.

## Key Features

- Side-by-side image comparison
- Automatic validation of image size (512x512)
- Image similarity analysis to enhance training results
- Image compression with convenient download option
- Generation of well-structured image datasets for efficient AI training

## Prerequisites

Before getting started, ensure that you have the following requirements:
- Python 3.10
- TailwindCSS
- PyTorch

## Installation 

To install the Image Comparison and Processing Tool, follow these steps:

1. Clone the repository:
git clone https://github.com/rliessum/image-comparison-tool.git
2. Navigate into the project directory:
cd image-comparison-tool
3. Install the dependencies:
pip install pipenv
pipenv --python 3.10
pipenv install

## Usage 

To use the tool, follow these steps:

1. Start the application:
pipenv shell
uvicorn main:app

This will start the application, you can access it ghoing to 127.0.0.1:8000 in your web browser.

## Contributing 

Contributions are always welcome! Please read the contribution guidelines first.

## License 

This project uses the following license: MIT.
