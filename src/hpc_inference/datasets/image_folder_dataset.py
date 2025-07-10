import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import concurrent.futures

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class ImageFolderDataset(Dataset):
    """
    Loads images from a folder, decodes with PIL, applies transform.
    - Handles common image extensions
    - Loads images as RGB by default (can be changed via color_mode)
    - Validates images using PIL if validate=True
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    @staticmethod
    def validate_PIL(file_path):
        """
        Validates if the file can be opened by PIL.
        Returns True if valid, False otherwise.
        """
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify that it is an image
            return True
        except (IOError, SyntaxError):
            return False
    
    @classmethod
    def validate_image_dir(cls, image_dir, max_workers=16):


        image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(cls.IMG_EXTS)
        ]

        if not image_files:
            raise ValueError(f"No valid image files found in directory: {image_dir}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(cls.validate_PIL, image_files))
        
        valid_files = [f for f, is_valid in zip(image_files, results) if is_valid]
        invalid_files = [f for f, is_valid in zip(image_files, results) if not is_valid]
        if invalid_files:
            logging.warning(f"Invalid image count: {len(invalid_files)}")
            logging.warning(f"Invalid image files: {invalid_files}")
        return valid_files

    def __init__(self, image_dir, transform=None, color_mode="RGB", validate=False):
        """
        Args:
            image_dir (str): Path to image folder.
            transform (callable, optional): Transform to apply to images.
            color_mode (str): Color mode for PIL.Image.convert (default: "RGB").
            validate (bool): If True, validates images in the directory using PIL.
                             If False, assumes all files are valid images.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.color_mode = color_mode

        if validate:
            self.image_files = sorted(self.validate_image_dir(image_dir))
            if not self.image_files:
                raise ValueError(f"No valid images found in directory: {image_dir}")
        else:
            self.image_files = sorted([
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith(self.IMG_EXTS)
            ])
       
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            img_path (str): Path to the image file.
            img (PIL.Image or Tensor): Image loaded in self.color_mode.
        """
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert(self.color_mode)
        if self.transform:
            img = self.transform(img)
        return img_path, img
