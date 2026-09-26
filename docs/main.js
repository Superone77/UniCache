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
