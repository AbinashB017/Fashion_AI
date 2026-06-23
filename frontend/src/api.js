import axios from 'axios';

// Read from Vercel environment variable, fallback to localhost for local dev
const API_BASE = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api` 
  : 'http://localhost:8000/api';

export const chatOutfits = async (query, gender_pref = "Any") => {
  const response = await axios.post(`${API_BASE}/chat`, { query, gender_pref });
  return response.data;
};

export const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post(`${API_BASE}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return response.data;
};

export const swapItem = async (current_outfit, slot_to_swap, occasion, gender, query) => {
  const response = await axios.post(`${API_BASE}/swap`, {
    current_outfit,
    slot_to_swap,
    occasion,
    gender,
    query
  });
  return response.data;
};
