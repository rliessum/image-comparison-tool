const uploadForm = document.getElementById("uploadForm");
const downloadButton = document.getElementById("downloadButton");
const downloadDone = document.getElementById("downloadDone");
const autoZoomButton = document.getElementById("autoZoomButton");
const spinner = document.getElementById("spinner");
const imageContainer = document.getElementById("imageContainer");
const matchedImageContainer = document.getElementById("matchedImageContainer");
const imagesSkeleton = document.getElementById("imagesSkeleton");

const fileInput = document.getElementById('fileUpload');
const progressElement = document.getElementById("progress");
const cleanOutputtedFilesButton = document.getElementById("cleanOutputtedFilesButton");
const postProccesFilesButton = document.getElementById('postProccesFilesButton');
const postProccesFilesText = document.getElementById('postProccesFilesText');
const postProccesFilesSpinner = document.getElementById('postProccesFilesSpinner');
const compareButton = document.getElementById("compareButton");
const compareText = document.getElementById("compareText");
const compareSpinner = document.getElementById("compareSpinner");

let processedImagesCount = 0;
let fileCount = 0;
let folderName = "";

const displayUploadedImages = (files, folder) => {
  imageContainer.innerHTML = "";
  processedImagesCount = 0;
  let filename = ""
  for (let file of files) {
    onImageProcessed();

    if (typeof file === "string") {
      let filename = file.split("/").pop();
      console.log(filename);

      const listItem = document.createElement("li");
      listItem.classList.add("aspect-h-1", "aspect-w-1", "relative", "rounded-md", "bg-gray-200", "lg:h-60");

      const imageElement = document.createElement("img");
      imageElement.src = `output/${folder}/512x512_ratio_raw/${filename}`;
      imageElement.classList.add("rounded-md", "image-border");
      // imageElement.addEventListener("load", () => {
      //   // Execute code after the image has loaded
      //   console.log(`Image loaded ${filename}`);
      //   createImageBorders(filename); // Add this line to create the image borders
      // });

      const filenameElement = document.createElement("p");
      filenameElement.textContent = filename; // Display the filename
      filenameElement.classList.add("hidden", "filename-text");

      listItem.appendChild(imageElement);
      listItem.appendChild(filenameElement);

      // Show filename on hover
      listItem.addEventListener("mouseenter", () => {
        filenameElement.classList.remove("hidden");
      });

      listItem.addEventListener("mouseleave", () => {
        filenameElement.classList.add("hidden");
      });

      imageContainer.appendChild(listItem);
    }
  }
};


const showSpinner = () => {
  spinner.classList.remove("hidden");
};

const hideSpinner = () => {
  spinner.classList.add("hidden");
};

const showSkeleton = () => {
  console.log("showSkeleton");
  imagesSkeleton.classList.remove("hidden");
};

const hideSkeleton = () => {
  console.log("hideSkeleton");
  imagesSkeleton.classList.add("hidden");
};

const removeResults = () => {
  imageContainer.innerHTML = "";
};

const downloadResults = async () => {
  try {
    const response = await fetch(`/download/${folderName}`);
    const timestamp = new Date().getTime();
    const isoTimestamp = new Date(timestamp).toISOString();

    if (response.ok) {
      const contentDisposition = response.headers.get("content-disposition");
      const fileName = contentDisposition
        ? contentDisposition.split("filename=")[1]
        : `result_${folderName}_${isoTimestamp}.zip`;

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", fileName);
      document.body.appendChild(link);
      link.click();

      URL.revokeObjectURL(url);
    } else {
      console.log("Download failed");
    }
  } catch (error) {
    console.error(error);
  }
};

const autoZoom = async () => {
  try {
    const response = await fetch(`/auto_zoom/${folderName}`);
    if (response.ok) {
      console.log("Auto Zoom completed");
    } else {
      console.error("Auto Zoom failed");
    }
  } catch (error) {
    console.error(error);
  }
};

downloadDone.addEventListener("click", async () => {
  if (folderName) {
    downloadResults();
    autoZoom();
  } else {
    console.error("No folder name available");
  }
});

compareButton.addEventListener("click", (e) => {
  console.log("Comparing images");
  compareSpinner.classList.remove("hidden");
  compareText.classList.add("hidden");
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(uploadForm);
  console.log("Upload images");
  removeResults();
  showSkeleton();
  showSpinner();

  try {
    const response = await fetch('/upload_and_process', {
      method: 'POST',
      body: formData
    });

    const reader = response.body.getReader();
    const contentLength = +response.headers.get('Content-Length');

    let receivedLength = 0;
    let chunks = [];

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      chunks.push(value);
      receivedLength += value.length;
      progressElement.classList.remove("hidden");
      const progressPercent = (receivedLength / contentLength) * 100;
      progressElement.value = progressPercent;
      progressElement.innerText = `${Math.round(progressPercent)}%`;
      progressElement.style.backgroundImage = `linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet ${progressPercent}%)`;
    }


    let chunksAll = new Uint8Array(receivedLength);
    let position = 0;
    for (let chunk of chunks) {
      chunksAll.set(chunk, position);
      position += chunk.length;
    }

    const blob = new Blob([chunksAll]);
    const text = await blob.text();
    const data = JSON.parse(text);
    compareSpinner.classList.add("hidden");
    compareText.classList.remove("hidden");
    hideSpinner();
    hideSkeleton();
    displayUploadedImages(uploadForm.files.files);

    if (response.ok) {
      folderName = data.folder_name;
      fileNames = data.filenames;
      files = fileNames;
      folder = folderName;
      console.log(files);
      console.log(folder);
      downloadButton.disabled = false;
      autoZoomButton.disabled = false;

      downloadButton.classList.add("hidden");
      downloadDone.classList.remove("hidden");
      console.log(data);

      // Process and display the uploaded images with borders
      displayUploadedImages(files, folder);
      createImageBorders(data); // Add this line
    } else {
      console.error("Upload failed");
    }
  } catch (error) {
    console.error(error);
  }
});

const createImageBorders = async (data) => {
  if (folderName) {
    const mostSimilarImages = data.most_similar_images;
    console.log(mostSimilarImages);
    if (Array.isArray(mostSimilarImages)) {
      // Add borders to the most similar image pairs
      for (let i = 0; i < mostSimilarImages.length; i++) {
        const [image1Path, image2Path] = mostSimilarImages[i];

        let image1Filename = image1Path.split("/").pop();
        let image2Filename = image2Path.split("/").pop();
        console.log(`Match ${image1Filename}, ${image2Filename}`);

        const pairContainer = document.createElement("li");
        pairContainer.classList.add("aspect-h-1", "aspect-w-2", "relative", "rounded-md", "bg-red-600", "flex", "px-2", "py-2", "w-full");

        const createImageElement = (src, filename) => {
          const listItem = document.createElement("div");
          listItem.classList.add("aspect-h-1", "aspect-w-1", "relative", "rounded-md", "bg-gray-200");

          const imageElement = document.createElement("img");
          imageElement.src = src;
          imageElement.classList.add("rounded-md", "image-border");

          const filenameElement = document.createElement("p");
          filenameElement.textContent = filename;
          filenameElement.classList.add("hidden", "filename-text");

          listItem.appendChild(imageElement);
          listItem.appendChild(filenameElement);

          // Show filename on hover
          listItem.addEventListener("mouseenter", () => {
            filenameElement.classList.remove("hidden");
          });

          listItem.addEventListener("mouseleave", () => {
            filenameElement.classList.add("hidden");
          });

          return listItem;
        }

        const imageElement1 = createImageElement(`output/${folderName}/512x512_ratio_raw/${image1Filename}`, image1Filename);
        const imageElement2 = createImageElement(`output/${folderName}/512x512_ratio_raw/${image2Filename}`, image2Filename);

        pairContainer.appendChild(imageElement1);
        pairContainer.appendChild(imageElement2);

        matchedImageContainer.appendChild(pairContainer);
      }
    } else {
      console.error("mostSimilarImages is not iterable");
    }
  } else {
    console.error("No folder name available");
  }
};


autoZoomButton.addEventListener("click", async () => {
  if (folderName) {
    autoZoom();
  } else {
    console.error("No folder name available");
  }
});

// listen for changes on the file input
fileInput.addEventListener('change', function () {
  // get the number of files
  fileCount = this.files.length; // Update fileCount here
  console.log(fileCount);
  // clear the skeleton container
  imagesSkeleton.innerHTML = ' ';

  // create a skeleton for each file
  for (let i = 0; i < fileCount; i++) {
    const skeleton = document.createElement('li');
    skeleton.className = "animate-pulse skeleton aspect-h-1 aspect-w-1 relative rounded-md bg-gray-200 lg:h-60"
    imagesSkeleton.appendChild(skeleton);
    compareText.innerText = `Ready to compare ${fileCount} images`;
  }
});



postProccesFilesButton.addEventListener('click', async () => {
  try {
    if (folderName) {
      console.log(`Post processing files for ${folderName}`);
      postProccesFilesSpinner.classList.remove("hidden");
      postProccesFilesText.classList.add("hidden");
      const procceedFiles = postProccesFiles(folderName);
    } else {
      throw new Error("No folder name available");
    }
  } catch (error) {
    console.error(error);
  }
});

async function postProccesFiles(folderName) {
  try {
    console.log(folderName);
    const response = await fetch(`/post_process/${folderName}`, {
      method: 'POST'
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    console.log(data.message);
    postProccesFilesSpinner.classList.add("hidden");
    postProccesFilesText.classList.remove("hidden");
  } catch (error) {
    console.error(error);
  }
}

async function cleanOutputtedFiles() {
  const response = await fetch('/clean_output', {
    method: 'POST'
  });
  const data = await response.json();
  console.log(data.message);
}

function confirmCleanOutputtedFiles() {
  if (confirm("Are you sure you want to clean the output directory?")) {
    cleanOutputtedFiles();
  }
}


const onImageProcessed = () => {
  processedImagesCount++;
  const progressPercent = (processedImagesCount / fileCount) * 100;
  progressElement.value = progressPercent;
  console.log(`Progress: ${progressPercent}%`);
  progressElement.innerText = `${Math.round(progressPercent)}%`;
  progressElement.style.backgroundImage = `linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet ${progressPercent}%)`;
  if (processedImagesCount === fileCount) {
    progressElement.classList.add("hidden");
    console.log(`${processedImagesCount}/${fileCount} images have been processed`);
  }
};