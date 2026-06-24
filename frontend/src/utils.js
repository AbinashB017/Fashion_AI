const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const getImageUrl = (itemId) => {
  if (!itemId) return "";
  if (itemId.includes("_")) {
    const [platform, fileId] = itemId.split("_");
    return `${BASE_URL}/images/${platform}/${fileId}.jpg`;
  }
  return `${BASE_URL}/images/${itemId}.jpg`;
};
