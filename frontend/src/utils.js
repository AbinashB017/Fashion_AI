export const getImageUrl = (itemId) => {
  if (!itemId) return "";
  if (itemId.includes("_")) {
    const [platform, fileId] = itemId.split("_");
    return `http://localhost:8000/images/${platform}/${fileId}.jpg`;
  }
  return `http://localhost:8000/images/${itemId}.jpg`;
};
