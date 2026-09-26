const button = document.getElementById("copy-command");
button?.addEventListener("click", async () => {
  const command = document.getElementById("inference-command")?.textContent;
  if (!command) return;
  try {
    await navigator.clipboard.writeText(command);
    button.textContent = "Copied";
    window.setTimeout(() => { button.textContent = "Copy"; }, 1600);
  } catch {
    button.textContent = "Select to copy";
  }
});

const galleryItems = [...document.querySelectorAll(".gallery-item")];
const galleryDialog = document.getElementById("gallery-dialog");
const galleryImage = document.getElementById("gallery-dialog-image");
const galleryTitle = document.getElementById("gallery-dialog-title");
const galleryPrompt = document.getElementById("gallery-dialog-prompt");
const galleryPosition = document.getElementById("gallery-position");
let galleryData = [];
let activeImage = 0;

fetch("./gallery.json")
  .then((response) => response.ok ? response.json() : [])
  .then((data) => { galleryData = data; })
  .catch(() => {});

function showGalleryImage(index) {
  activeImage = (index + galleryItems.length) % galleryItems.length;
  const item = galleryItems[activeImage];
  const image = item.querySelector("img");
  galleryImage.src = image.src;
  galleryImage.alt = image.alt;
  galleryTitle.textContent = galleryData[activeImage]?.title || item.querySelector("figcaption")?.textContent.trim() || "Image generation";
  galleryPrompt.textContent = galleryData[activeImage]?.prompt || "";
  galleryPosition.textContent = `${activeImage + 1} / ${galleryItems.length}`;
}

galleryItems.forEach((item, index) => {
  const openButton = document.createElement("button");
  openButton.type = "button";
  openButton.className = "gallery-open";
  openButton.setAttribute("aria-label", `View ${item.querySelector("img").alt}`);
  openButton.title = "View image and prompt";
  openButton.addEventListener("click", () => {
    showGalleryImage(index);
    galleryDialog.showModal();
  });
  item.append(openButton);
});

document.getElementById("gallery-close")?.addEventListener("click", () => galleryDialog.close());
document.getElementById("gallery-prev")?.addEventListener("click", () => showGalleryImage(activeImage - 1));
document.getElementById("gallery-next")?.addEventListener("click", () => showGalleryImage(activeImage + 1));
galleryDialog?.addEventListener("click", (event) => {
  if (event.target === galleryDialog) galleryDialog.close();
});
galleryDialog?.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") showGalleryImage(activeImage - 1);
  if (event.key === "ArrowRight") showGalleryImage(activeImage + 1);
});
