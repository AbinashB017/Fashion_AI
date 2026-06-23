import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, Loader2 } from 'lucide-react';
import clsx from 'clsx';

export default function UploadZone({ onUpload, isLoading }) {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      onUpload(file);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {'image/*': []},
    multiple: false
  });

  return (
    <div 
      {...getRootProps()} 
      className={clsx(
        "relative flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-xl transition-all cursor-pointer overflow-hidden",
        isDragActive ? "border-purple-500 bg-purple-500/10" : "border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/50",
        preview ? "border-none" : ""
      )}
    >
      <input {...getInputProps()} />
      
      {preview && (
        <div className="absolute inset-0 w-full h-full">
          <img src={preview} alt="Upload preview" className="object-cover w-full h-full opacity-40 blur-sm" />
          <img src={preview} alt="Upload preview crisp" className="absolute inset-0 w-full h-full object-contain" />
        </div>
      )}

      {isLoading ? (
        <div className="z-10 flex flex-col items-center p-4 bg-zinc-900/80 rounded-lg backdrop-blur-md">
          <Loader2 className="w-8 h-8 text-purple-400 animate-spin mb-2" />
          <p className="text-zinc-200 font-medium">Styling your item...</p>
        </div>
      ) : (
        !preview && (
          <div className="z-10 flex flex-col items-center">
            <UploadCloud className="w-12 h-12 text-zinc-400 mb-4" />
            <p className="text-zinc-200 font-medium">Drag & drop an item of clothing</p>
            <p className="text-zinc-500 text-sm mt-1">or click to browse files</p>
          </div>
        )
      )}
    </div>
  );
}
