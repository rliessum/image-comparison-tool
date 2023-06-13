import os
import shutil
import uuid
from tempfile import TemporaryDirectory
from typing import List
import cv2
import shutil
import torch
import logging
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from retinaface import RetinaFace
import tensorflow as tf
from PIL import Image

# Create a new FastAPI application
app = FastAPI()

# Set up CORS middleware to allow requests from any origin
origins = (["*"],)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(
    level=logging.CRITICAL, format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
logging.getLogger("").addHandler(console_handler)

# Load the pre-trained YOLOv5 model with force_reload=True
device = torch.device(0 if torch.cuda.is_available() else "cpu")
model = torch.hub.load(
    "ultralytics/yolov5", "yolov5x", device=device, force_reload=False
)
# model = torch.hub.load(
#     "ultralytics/yolov5", "yolov5x6", device=device, force_reload=False
# )

# Define the aspect ratios to consider
aspect_ratios = [
    (512, 512),
    (512, 768),
    (768, 512),
    (640, 960),
    (960, 640),
    (768, 1024),
    (1024, 768),
]

# Create a Jinja2Templates instance for the templates directory
templates = Jinja2Templates(directory="templates")

# Mount the static files directory for serving CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")

executor = ThreadPoolExecutor(max_workers=4)  # Adjust the max_workers as necessary


async def read_image(image_path: str) -> np.ndarray:
    """
    Read an image file using cv2.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image as a numpy array.
    """
    img = cv2.imread(image_path)
    return img


def run_model(img: np.ndarray) -> np.ndarray:
    """
    Run the image through the model and return the results.

    Args:
        img (numpy.ndarray): The input image as a numpy array.

    Returns:
        numpy.ndarray: The results of running the image through the model.
    """
    results = model(img)
    return results


def save_image(image: Image, output_path: str) -> None:
    """
    Save an image to a file.

    Args:
        image (PIL.Image): The image to save.
        output_path (str): The path to save the image to.
    """
    with open(output_path, "wb") as f:
        image.save(f)


def find_human_in_image(img):
    """
    Run the image through the model and find the first human detected in the image.

    Args:
        img (numpy.ndarray): The input image as a numpy array.

    Returns:
        Tuple[float, float, float, float, float]: A tuple containing the coordinates of the bounding box for the first human detected in the image.
    """
    results = model(img)
    human = next((x for x in results.xyxy[0] if int(x[5]) == 0), None)
    return human


def adjust_image_ratio(x1, y1, x2, y2, d1, d2, img, target_ratio, current_ratio):
    """
    Adjust the image ratio by expanding the bounding box.

    Args:
        x1 (int): The x-coordinate of the top-left corner of the bounding box.
        y1 (int): The y-coordinate of the top-left corner of the bounding box.
        x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (int): The y-coordinate of the bottom-right corner of the bounding box.
        d1 (float): The distance between the left eye and the nose.
        d2 (float): The distance between the right eye and the nose.
        img (numpy.ndarray): The input image.
        target_ratio (float): The target aspect ratio.
        current_ratio (float): The current aspect ratio.

    Returns:
        Tuple[int, int, int, int, int]: A tuple containing the adjusted bounding box coordinates and the number of iterations performed.
    """
    ratio_threshold = 0.9997
    loop_counter = 0
    while True:
        desired_ratio = (
            current_ratio / target_ratio
            if current_ratio > target_ratio
            else target_ratio / current_ratio
        )
        if (desired_ratio > ratio_threshold) or loop_counter > 2000:
            break
        if current_ratio < target_ratio:
            x1 = max(0, x1 - 1)
            x2 = min(img.shape[1] - 1, x2 + 1)
        elif current_ratio > target_ratio:
            y1 = max(0, y1 - 1)
            y2 = min(img.shape[0] - 1, y2 + 1)
        current_width = x2 - x1
        current_height = y2 - y1
        current_ratio = current_width / current_height
        loop_counter += 1
    return x1, y1, x2, y2, loop_counter


def auto_zoom(images: List[str], output_base_dir: str):
    """
    Perform auto zoom on the specified images and save the results to the specified output directory.

    Args:
        images (List[str]): A list of paths to the images to process.
        output_base_dir (str): The base directory to save the processed images to.

    Returns:
        None
    """
    logging.info("Starting auto-zoom...")
    # loop over all files in the input directory
    for image_path in images:
        # create the full input path and read the file
        img = cv2.imread(image_path)

        if img is None:
            continue
        height, width, _ = img.shape

        # Assign the image bounds to c1, c2, d1, d2
        c1, c2, d1, d2 = 0, 0, width, height

        # Run the image through the model
        results = model(img)

        # Find the first human detected in the image
        human = next((x for x in results.xyxy[0] if int(x[5]) == 0), None)

        if human is None:
            print(f"No human detected in the image {image_path}.")
            os.remove(image_path)
            continue

        # Crop the image to the bounding box of the human
        x1, y1, x2, y2 = map(int, human[:4])
        orgx1, orgx2, orgy1, orgy2 = x1, y1, x2, y2
        orig_width, orig_height = x2 - x1, y2 - y1

        for ratio in aspect_ratios:
            width, height = ratio
            x1, y1, x2, y2 = orgx1, orgx2, orgy1, orgy2
            target_ratio = width / height
            current_width = x2 - x1
            current_height = y2 - y1
            current_ratio = current_width / current_height

            def within_bounds(x1, y1, x2, y2):
                return x1 >= 0 and y1 >= 0 and x2 < d1 and y2 < d2

            ratio_threshold = 0.9997
            while True:
                loop_counter = 0
                desired_ratio = current_ratio / target_ratio
                if current_ratio > target_ratio:
                    desired_ratio = target_ratio / current_ratio
                if desired_ratio > ratio_threshold:
                    break
                while desired_ratio < ratio_threshold:
                    if desired_ratio >= ratio_threshold:
                        break
                    if loop_counter > 2000:
                        break
                    loop_counter += 1

                    if current_ratio < target_ratio:
                        if x1 > 0:
                            x1 -= 1
                        if x2 < img.shape[1] - 1:
                            x2 += 1
                    elif current_ratio > target_ratio:
                        if y1 > 0:
                            y1 -= 1
                        if y2 < img.shape[0] - 1:
                            y2 += 1

                    current_width = x2 - x1
                    current_height = y2 - y1
                    current_ratio = current_width / current_height
                    desired_ratio = current_ratio / target_ratio
                    if current_ratio > target_ratio:
                        desired_ratio = target_ratio / current_ratio

                if within_bounds(x1, y1, x2, y2) and loop_counter <= 2000:
                    break

                ratio_threshold -= 0.005
                if ratio_threshold < 0:
                    x1, y1, x2, y2 = orgx1, orgx2, orgy1, orgy2
                    break

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # print(
            #     f"Final coords: {(x1, y1, x2, y2)}, Final ratio: {current_ratio}, Loops: {loop_counter}"
            # )

            # Crop the image
            cropped_img = img[y1:y2, x1:x2]

            # Convert BGR image to RGB for PIL
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

            # Convert array to Image for visualization
            result_img = Image.fromarray(cropped_img_rgb)

            # Create a directory for each aspect ratio if it doesn't exist
            output_dir = os.path.join(output_base_dir, f"{width}x{height}_ratio_raw")
            os.makedirs(output_dir, exist_ok=True)

            # create the full output path and write the file
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            result_img.save(output_path)


def move_image(org_image_path, target_folder):
    if not os.path.exists(org_image_path):
        return False
    image_filename = os.path.basename(org_image_path)
    new_image_path = os.path.join(target_folder, image_filename)
    shutil.move(org_image_path, new_image_path)


def move_copy(org_image_path, target_folder):
    if not os.path.exists(org_image_path):
        return False
    image_filename = os.path.basename(org_image_path)
    new_image_path = os.path.join(target_folder, image_filename)
    shutil.copy(org_image_path, new_image_path)


def detect_and_move_image(
    org_image_path,
    target_folder_post_processed,
    target_folder_multiple,
    target_folder_low_colors,
    target_folder_0,
):
    faces = RetinaFace.detect_faces(org_image_path)
    num_faces = len(faces)
    # print("There are", num_faces, "faces in the image")

    if num_faces > 1:
        move_copy(org_image_path, target_folder_multiple)
    if num_faces < 1:
        move_copy(org_image_path, target_folder_0)
    if count_colors(org_image_path) < 20000:
        move_copy(org_image_path, target_folder_low_colors)

    move_copy(org_image_path, target_folder_post_processed)


def count_colors(image_path):
    if not os.path.exists(image_path):
        return 0

    with Image.open(image_path) as image:
        # Convert the image to RGB mode if it's not already
        image = image.convert("RGB")

        # Get the size of the image
        width, height = image.size

        # Create an empty set to store unique colors
        colors = set()

        # Iterate over each pixel in the image
        for x in range(width):
            for y in range(height):
                # Get the RGB values of the pixel
                r, g, b = image.getpixel((x, y))

                # Add the RGB values as a tuple to the set
                colors.add((r, g, b))

        # Return the number of unique colors
        return len(colors)


@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/beta")
def serve_beta(request: Request):
    return templates.TemplateResponse("compare_images.html", {"request": request})


@app.post("/compare")
async def compare_images(files: List[UploadFile] = File(...)):
    """
    Compare the images uploaded in the request and perform auto zoom on them.

    Args:
        files (List[UploadFile]): A list of UploadFile objects containing the images to compare.

    Returns:
        dict: A dictionary containing the name of the folder containing the processed images.
    """
    logging.info("Received request to compare images")
    with TemporaryDirectory() as temp_dir:
        images = []
        for file in files:
            unique_filename = f"{temp_dir}/{str(uuid.uuid4())}.png"
            with open(unique_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            images.append(unique_filename)
        folder_name = str(uuid.uuid4())
        output_dir = os.path.join("output", folder_name)
        os.makedirs(output_dir, exist_ok=True)
        auto_zoom(images, output_dir)
    return {"folder_name": folder_name, "filenames": images}


@app.get("/download/{folder_name}")
async def download_results(folder_name: str):
    """
    Download the results of the image processing for the specified folder.

    Args:
        folder_name (str): The name of the folder containing the processed images.

    Returns:
        FileResponse: A FileResponse object containing the ZIP archive of the processed images.
    """
    logging.info(f"Downloading results for {folder_name}")
    folder_path = os.path.join("output", folder_name)
    zip_path = os.path.join("output", f"{folder_name}.zip")
    shutil.make_archive(folder_path, "zip", folder_path)
    return FileResponse(zip_path, media_type="application/zip")


@app.get("/auto_zoom/{folder_name}")
async def perform_auto_zoom(folder_name: str):
    """
    Perform auto zoom on the images in the specified folder.

    Args:
        folder_name (str): The name of the folder containing the images to process.

    Returns:
        dict: A dictionary containing a message indicating that the auto zoom process has completed.
    """
    logging.info(f"Running auto_zoom on {folder_name}")
    folder_path = os.path.join("output", folder_name)
    images = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    output_base_dir = os.path.join("output", f"{folder_name}_auto_zoom")
    os.makedirs(output_base_dir, exist_ok=True)
    auto_zoom(images, output_base_dir)
    return {"message": "Auto Zoom completed"}


@app.post("/post_process/{folder_name}")
async def process_images(folder_name: str, output_dir: str = None):
    """
    Process images in the specified folder and copy images with exactly one face to the output directory.

    Args:
        folder_name (str): The name of the folder containing the images to process.
        output_dir (str, optional): The path to the output directory. If not specified, a default value will be used.

    Returns:
        None
    """
    logging.info(f"Running post_process on {folder_name}")
    if not output_dir:
        output_dir = f"output/{folder_name}/512x512_ratio_raw"
        logging.info(f"Using {output_dir} as output directory.")
    logging.info(f"Running post_process on {folder_name}")

    # Set the GPU device
    logging.info(tf.test.gpu_device_name())

    # Find the latest directory based on its creation time
    # latest_dir = max(glob.glob(os.path.join(output_dir, "*/")), key=os.path.getctime)

    # Print the path to the latest directory
    logging.info(f"Using {output_dir} as input directory.")

    original_images_folder = Path(output_dir) / "512x512_ratio_raw"
    target_folder_post_processed = Path(output_dir) / "processed"
    target_folder_multiple = Path(output_dir) / "a1_more_faces"
    target_folder_0 = Path(output_dir) / "a1_0_faces"
    target_folder_low_colors = Path(output_dir) / "a1_low_colors"

    folders_to_create = [
        original_images_folder,
        target_folder_post_processed,
        target_folder_multiple,
        target_folder_0,
        target_folder_low_colors,
    ]

    for folder in folders_to_create:
        folder.mkdir(parents=True, exist_ok=True)

    # Counter variable to keep track of the number of images processed
    image_counter = 0

    # Get the list of image filenames
    image_filenames = [f for f in original_images_folder.glob("*") if f.is_file()]

    # Iterate through each image in the original images folder

    for image_path in image_filenames:
        try:
            detect_and_move_image(
                str(image_path),
                str(target_folder_post_processed),
                str(target_folder_multiple),
                str(target_folder_low_colors),
                str(target_folder_0),
            )
            image_counter += 1
            logging.info(f"Total images processed: {image_counter}")
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")

    logging.info(
        {
            "output_dir": str(output_dir),
            "message": "Image processing completed",
            "Total images processed": image_counter,
        }
    )
    return JSONResponse(
        {
            "output_dir": str(output_dir),
            "message": "Image processing completed",
            "Total images processed": image_counter,
        }
    )


@app.post("/clean_output")
async def clean_output():
    output_dir = os.path.join(os.getcwd(), "output")
    print(f"Cleaning output directory for {output_dir}")
    if not os.path.exists(output_dir):
        return {"message": "Output directory does not exist."}
    for dirpath, dirnames, filenames in os.walk(output_dir, topdown=False):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except PermissionError as pe:
                print(
                    f"Permission error deleting file: {file_path}. Error details: {pe}"
                )
            except OSError as oe:
                print(
                    f"OS error (file might be open or locked): {file_path}. Error details: {oe}"
                )
            except Exception as e:
                print(f"Error deleting file: {file_path}. Error details: {e}")
        for dirname in dirnames:
            dir_path = os.path.join(dirpath, dirname)
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error deleting directory: {dir_path}. Error details: {e}")
    return {"message": f"{output_dir} directory and its subdirectories cleaned."}


def within_bounds(x1, y1, x2, y2, d1, d2):
    return x1 >= 0 and y1 >= 0 and x2 < d1 and y2 < d2


async def save_upload_file(upload_file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()


# Use type hints for function arguments and return values
async def process_image(image_path: str) -> str:
    """
    Process an image by running it through the YOLOv5 model and saving the results.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        str: The path to the output image file.
    """
    # Read the input image
    img = await read_image(image_path)

    # Run the image through the model
    results = await asyncio.get_event_loop().run_in_executor(executor, run_model, img)

    # Find the first human detected in the image
    human = next((x for x in results.xyxy[0] if int(x[5]) == 0), None)

    if human is None:
        raise HTTPException(status_code=400, detail="No human detected in the image.")

    # Adjust the aspect ratio of the bounding box
    x1, y1, x2, y2 = human[:4]
    d1 = np.linalg.norm(results.landmarks[0][0] - results.landmarks[0][2])
    d2 = np.linalg.norm(results.landmarks[0][1] - results.landmarks[0][3])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target_ratio = 1.0
    current_ratio = (x2 - x1) / (y2 - y1)
    x1, y1, x2, y2, _ = RetinaFace.adjust_image_ratio(
        x1, y1, x2, y2, d1, d2, img, target_ratio, current_ratio
    )

    # Crop the image to the bounding box
    img = img[y1:y2, x1:x2]

    # Save the output image
    output_path = f"output/{uuid.uuid4()}.jpg"
    await asyncio.get_event_loop().run_in_executor(
        executor, save_image, Image.fromarray(img), output_path
    )

    return output_path


# Define the API endpoints
@app.post("/api/process-image")
async def api_process_image(
    request: Request, file: UploadFile = File(...)
) -> JSONResponse:
    """
    Process an uploaded image by running it through the YOLOv5 model and returning the path to the output image file.

    Args:
        request (fastapi.Request): The incoming request.
        file (fastapi.UploadFile): The uploaded image file.

    Returns:
        fastapi.JSONResponse: The path to the output image file.
    """
    # Save the uploaded image to a temporary file
    with TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Process the image
        try:
            output_path = await process_image(file_path)
        except HTTPException as e:
            return e.detail

        # Return the path to the output image file
        return JSONResponse({"path": output_path})


async def calculate_suitable_image(images, output_dir, target_size=(512, 512)):
    """
    Calculate a suitable image for AI model learning of humans.
    """
    logging.info(f"Calculating suitable image for AI model learning of humans.")
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLOv5 model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    logging.info(f"Loading YOLOv5 model.")
    # check what model is loaded
    # logging.info(f"Model: {model}")
    # Process each input image
    for i, image_path in enumerate(images):
        # Read the input image
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Image {image_path} could not be read.")
            continue

        # Detect faces in the image using YOLOv5
        results = model(img)

        # Extract the bounding boxes of the detected faces
        bboxes = results.xyxy[0][:, :4].cpu().numpy()

        # Crop the image to the bounding boxes of the detected faces
        for j, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.astype(int)
            face_img = img[y1:y2, x1:x2]

            # Save the face image to the output directory
            output_path = os.path.join(output_dir, f"{i}_{j}.jpg")
            cv2.imwrite(output_path, face_img)

    logging.info(f"Suitable images saved to {output_dir}.")


async def process_and_compare_images(images, output_dir, similarity_threshold=50):
    """
    Compare all pairs of images in the input list.
    If two images are similar (difference_percentage > similarity_threshold),
    add their paths to the list of similar images.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a dictionary to store images
    images_dict = {}
    for i, image in enumerate(images):
        img = cv2.imread(image)
        if img is None:
            logging.warning(f"Image {image} could not be read.")
            continue
        images_dict[i] = img

    difference_percentages = {}
    for i, image1 in images_dict.items():
        for j, image2 in images_dict.items():
            if j <= i:
                continue
            try:
                # Check image shapes and channels
                if image1.shape == image2.shape and image1.ndim == image2.ndim:
                    # Compute the absolute difference between the images
                    difference = cv2.absdiff(image1, image2)
                    total_possible_difference = np.prod(image1.shape) * 255

                    # Compute the sum of the differences
                    total_difference = np.sum(difference)
                    difference_percentage = (
                        total_difference / total_possible_difference
                    ) * 100

                    difference_percentages[
                        (images[i], images[j])
                    ] = difference_percentage

                    # Check if images are similar
                    if difference_percentage > similarity_threshold:
                        logging.info(
                            f"Images {images[i]} and {images[j]} are similar: {difference_percentage}%"
                        )
                    else:
                        logging.info(
                            f"Images {images[i]} and {images[j]} are not similar: {difference_percentage}%"
                        )
            except Exception as e:
                logging.error(
                    f"Error while comparing image {images[i]} with {images[j]}: {e}"
                )

    return difference_percentages


def get_most_similar_images(difference_percentages, num_images=2):
    # Sort the dictionary by difference percentage in ascending order
    sorted_diff_percentages = sorted(difference_percentages.items(), key=lambda x: x[1])

    # Get the paths of the most similar images
    most_similar_images = [pair for pair, _ in sorted_diff_percentages[:num_images]]

    return most_similar_images


@app.post("/upload_and_process")
async def upload_and_process(files: List[UploadFile] = File(...)):
    logging.info(f"Files received: {files}")
    with TemporaryDirectory() as temp_dir:
        images = []

        for file in files:
            unique_filename = f"{temp_dir}/{str(uuid.uuid4())}.png"
            await save_upload_file(file, unique_filename)
            images.append(unique_filename)
        logging.info(f"Images saved: {images}")

        folder_name = str(uuid.uuid4())
        output_dir = os.path.join("output", folder_name)
        logging.info(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory created: {output_dir}")
        auto_zoom(images, output_dir)
        logging.info(f"Auto zoom completed: {output_dir}")
        proccesed_images = await process_images(folder_name, output_dir)

        suitable_images = await calculate_suitable_image(
            images, output_dir, target_size=(224, 224)
        )
        # Process all images and compare them
        difference_percentages = await process_and_compare_images(images, output_dir)
        logging.info(f"Images processed and compared: {output_dir}")
        # Get the paths of the most similar images
        most_similar_images = get_most_similar_images(
            difference_percentages, num_images=5
        )
        logging.info(f"Most similar images: {most_similar_images}")
        # Return the most similar images in the response
        return JSONResponse(
            {
                "folder_name": folder_name,
                "filenames": images,
                "difference_percentages": {
                    f"{pair[0]} and {pair[1]}": str(round(percentage, 2))
                    for pair, percentage in difference_percentages.items()
                },
                "most_similar_images": most_similar_images,
            }
        )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
