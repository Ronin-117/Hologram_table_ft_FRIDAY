const images = document.querySelectorAll(".draggable");
let activeImage = null;
let offsetX = 0;
let offsetY = 0;

// Function to handle clicking on the image
images.forEach((image) => {
  image.addEventListener("click", function (e) {
    if (!activeImage) {
      // If no image is active, lock this image
      activeImage = image;

      // Get the bounding box of the image to calculate the offset correctly
      const rect = image.getBoundingClientRect();

      // Calculate the offset relative to the current mouse position and image's top-left corner
      offsetX = e.clientX - rect.left;
      offsetY = e.clientY - rect.top;

      image.style.position = "absolute"; // Ensure position is absolute for dragging
      image.style.zIndex = 1000; // Bring image to the front
    } else if (activeImage === image) {
      // If this image is already active, unlock it (second click)
      activeImage = null;
      image.style.zIndex = 1; // Reset the z-index when unlocking
    }
  });
});

// Function to move the image with the mouse pointer
window.addEventListener("mousemove", function (e) {
  if (activeImage) {
    // Move the image according to the mouse position, adjusting for the offset
    activeImage.style.left = e.clientX - offsetX + "px";
    activeImage.style.top = e.clientY - offsetY + "px";
  }
});

// Optional: Allow unlocking by clicking anywhere on the window
window.addEventListener("click", function (e) {
  // Unlock image if there's an active one, and it's not the clicked image
  if (activeImage && !e.target.classList.contains("draggable")) {
    activeImage.style.zIndex = 1; // Reset z-index
    activeImage = null; // Release the image
  }
});
