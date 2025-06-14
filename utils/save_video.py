import os
import imageio
import re

def save_video(folder, output_filename, fps=30):
    def extract_index(filename):
        # Extract digits following 'full'
        match = re.search(r"full(\d+)", filename)
        if match:
            return int(match.group(1))
        return -1

    images = []
    # Filter and sort files based on the index extracted from the filename.
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png') and f.startswith("full")],
                   key=extract_index)
    for filename in files:
        img_path = os.path.join(folder, filename)
        images.append(imageio.v2.imread(img_path))
    imageio.mimsave(output_filename, images, fps=fps)

def save_video_hd(folder, output_filename, fps=30, codec='libx264', bitrate='5000k'):
    """
    Save a high-definition video from PNG images in the given folder.
    
    Images must have filenames starting with "full" and ending with ".png", with an index 
    (e.g., "full123.png"). This function sorts the images by the index extracted from the filename.
    
    The video is saved using the specified codec and bitrate for higher quality.
    
    Args:
        folder (str): Path to the folder containing the images.
        output_filename (str): Path for the output video file (e.g., "output.mp4").
        fps (int): Frames per second for the video.
        codec (str): Codec to use (default: 'libx264').
        bitrate (str): Bitrate for the video (default: '5000k').
    
    Returns:
        None
    """
    def extract_index(filename):
        match = re.search(r"full(\d+)", filename)
        if match:
            return int(match.group(1))
        return -1

    # Sort files based on the extracted index.
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith('.png') and f.startswith("full")],
        key=extract_index
    )
    
    writer = imageio.get_writer(
        output_filename, 
        fps=fps, 
        codec=codec,
        bitrate=bitrate
    )
    
    for filename in files:
        img_path = os.path.join(folder, filename)
        image = imageio.imread(img_path)
        writer.append_data(image)
    
    writer.close()
