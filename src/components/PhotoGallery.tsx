import React from 'react';
import { ImageRecord } from '../services/NeRFDatabaseService';

interface PhotoGalleryProps {
  images: ImageRecord[];
  onImageSelect: (imageId: string) => void;
}

export const PhotoGallery: React.FC<PhotoGalleryProps> = ({ images, onImageSelect }) => {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 max-h-96 overflow-y-auto p-2 border border-gray-300 rounded-md">
      {images.length === 0 ? (
        <p className="col-span-full text-center text-gray-500">No images uploaded yet.</p>
      ) : (
        images.map((image) => {
          const objectURL = URL.createObjectURL(image.data);
          return (
            <div
              key={image.id}
              className="relative w-full aspect-square cursor-pointer overflow-hidden rounded-md shadow-sm hover:shadow-md transition-shadow"
              onClick={() => onImageSelect(image.id)}
            >
              <img
                src={objectURL}
                alt={image.filename}
                className="w-full h-full object-cover"
                onLoad={() => URL.revokeObjectURL(objectURL)} // Revoke URL after image loads
                onError={() => URL.revokeObjectURL(objectURL)} // Revoke URL on error too
              />
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-1 text-white text-xs truncate">
                {image.filename}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
};
