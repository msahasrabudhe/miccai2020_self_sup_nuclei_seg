__author__ = 'marvinler'
import imageio
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import shutil
from concurrent import futures
from oslide.examples.deepzoom import deepzoom_tile

def download_svs(source_folder, gdc_executable_path):
    # Retrieve files of the input folder, and seek the manifest file to be processed by GDC tool
    fs = list(map(lambda x: os.path.abspath(os.path.join(source_folder, x)), os.listdir(source_folder)))
    files = [f for f in fs if os.path.isfile(f)]
    manifest_file = [f for f in files if 'manifest' in f.lower()]
    assert len(manifest_file) == 1, 'multiple manifest files found'
    manifest_file = manifest_file[0]

    # Compute output dir: same as the one containing the manifest file
    output_dir = os.path.dirname(manifest_file)

    # Verify that the GDC path is correct
    assert os.path.exists(gdc_executable_path), "GDC executable not found at %s" % gdc_executable_path

    # Opens manifest and retrieves the ids to be retrieved
    with open(manifest_file, 'r') as f:
        lines = f.read().splitlines()
    lines = lines[1:]
    ids = [line.split('\t')[0] for line in lines]
    filenames = [line.split('\t')[1] for line in lines]
    md5sums = [line.split('\t')[2] for line in lines]

    for file_id in tqdm(ids):
        if os.path.exists(os.path.join(output_dir, file_id)):
            continue
        subprocess.check_output([gdc_executable_path, 'download', '--dir', output_dir,
                                 '--n-processes', '4', file_id])

    return [os.path.abspath(os.path.join(os.path.join(output_dir, id_file), filename)) for id_file, filename in
            zip(ids, filenames)], md5sums

def slide2tiles_withargs(svs_file, OUT_ROOT, tile_width, label, **kwargs):
    output_start = os.path.join(OUT_ROOT, label, os.path.splitext(os.path.basename(svs_file))[0])

    if not os.path.exists(output_start):
        print('  processing', output_start)

        basename        = output_start
        formats         = 'png' if formats not in kwargs else kwargs['formats']
        tile_size       = tile_width
        overlap         = 0 if 'overlap' not in kwargs else kwargs['overlap']
        limit_bounds    = True  if 'limit_bounds' not in kwargs else kwargs['limit_bounds']
        quality         = 90 if 'quality' not in kwargs else kwargs['quality']
        workers         = 6  if 'workers' not in kwargs else kwargs['workers']
        with_viewer     = True
        deepzoom_tile.DeepZoomStaticTiler(svs_file, basename, formats, 
                tile_size, overlap, limit_bounds, quality, workers, with_viewer).run()
        # python3 -m oslide.examples.deepzoom.deepzoom_tile -e 0 -j 6 -s tile_width -r -o output_start svs_file

    return os.path.join(output_start, 'slide_files/')


def slide2tiles(svs_file, OUT_ROOT, tile_width, output_start):

    if not os.path.exists(output_start):
        print('  processing', output_start)

        basename        = output_start
        formats         = 'png'
        tile_size       = tile_width
        overlap         = 8
        limit_bounds    = True
        quality         = 90
        workers         = 6
        with_viewer     = True
        deepzoom_tile.DeepZoomStaticTiler(svs_file, basename, formats, 
                tile_size, overlap, limit_bounds, quality, workers, with_viewer).run()
        # python3 -m oslide.examples.deepzoom.deepzoom_tile -e 0 -j 6 -s tile_width -r -o output_start svs_file

    return os.path.join(output_start, 'slide_files/')

def extract_tiles(svs_filepath, tile_width=228, formats='png', overlap=0, limit_bounds=True,  quality=90, workers=6, with_viewer=False):
    output_start = os.path.splitext(os.path.basename(svs_filepath))[0]

    if not os.path.exists(output_start):
        print('  processing', output_start)
        deepzoom_tile.DeepZoomStaticTiler(svs_filepath, output_start, formats, 
                tile_width, overlap, limit_bounds, quality, workers, with_viewer).run()
    else:
        print('%s already exists. ' %(output_start))
    return os.path.join(output_start, 'slide_files/')


def slides2tiles(svs_files, tile_width):
    for i, svs_file in enumerate(svs_files):
        print('  %d/%d' % (i, len(svs_files) - 1))
        yield slide2tiles(svs_file, tile_width)


def select_level(tiles_folders):
    res = []
    for tiles_folder in tiles_folders:
        folders = [os.path.join(tiles_folder, f) for f in os.listdir(tiles_folder) if not os.path.isfile(f)]
        folders = sorted(folders, key=lambda v: int(os.path.basename(v)))
        res.append(folders[-1])
    return res


def _compute_pixelwise_is_background(jpeg_file, background_pixel_value):
    img = imageio.imread(jpeg_file)

    channel_above_threshold = img > background_pixel_value
    pixel_above_threshold = np.prod(channel_above_threshold, axis=-1)

    return img, pixel_above_threshold


def save_color_background(jpeg_file, background_pixel_value, output_dir, output_filename=None):
    img, pixel_above_threshold = _compute_pixelwise_is_background(jpeg_file, background_pixel_value)

    colored_img = copy.deepcopy(img)
    colored_img[:, :, 0] = np.where(pixel_above_threshold, 0, img[:, :, 0])
    colored_img[:, :, 1] = np.where(pixel_above_threshold, 0, img[:, :, 1])
    colored_img[:, :, 2] = np.where(pixel_above_threshold, 255, img[:, :, 2])

    concatenated_image = np.concatenate((img, colored_img), axis=1)
    output_filename = os.path.basename(jpeg_file) if output_filename is None else output_filename
    output_filename = '%.3f' % (np.sum(pixel_above_threshold) / (img.shape[0] * img.shape[1])) + '__' + output_filename
    imageio.imsave(os.path.join(output_dir, output_filename), concatenated_image)


def is_tile_mostly_background(jpeg_file, background_pixel_value, background_threshold, expected_shape):
    """ Returns True if tile percent of background pixels are above background_threshold or tile is not of shape
    expected_shape.

    :param jpeg_file: abs path to jpeg image
    :param background_pixel_value: threshold above which a channel pixel is considered background
    :param background_threshold: percent above which a tile is considered backgrond based on is pixel background
    :param expected_shape: expected shape of tile
    """
    img, pixel_above_threshold = _compute_pixelwise_is_background(jpeg_file, background_pixel_value)
    if img.shape != expected_shape:
        return True, 0.

    percent_background_pixels = np.sum(pixel_above_threshold) / (img.shape[0] * img.shape[1])
    return percent_background_pixels > background_threshold, percent_background_pixels


def remove_unspecified_slides_folders(source_folder):
    # Retrieve files of the input folder, and seek the manifest file to be processed by GDC tool
    """ Helper to remove downloaded slides folders that are not specified within the associated manifest file.

    :param source_folder: folder containing the subfolders of SVS file (one subfolder per SVS file)
    """
    fs = list(map(lambda x: os.path.abspath(os.path.join(source_folder, x)), os.listdir(source_folder)))
    files = [f for f in fs if os.path.isfile(f)]
    manifest_file = [f for f in files if 'manifest' in f.lower()]
    assert len(manifest_file) == 1, 'multiple manifest files found'
    manifest_file = manifest_file[0]

    # Compute output dir: same as the one containing the manifest file
    output_dir = os.path.dirname(manifest_file)

    dirs = [f for f in os.listdir(source_folder) if
            not os.path.isfile(os.path.join(source_folder, f)) and not os.path.join(source_folder, f).startswith('gdc')]
    print(dirs)

    # Opens manifest and retrieves the ids to be retrieved
    with open(manifest_file, 'r') as f:
        lines = f.read().splitlines()
    lines = lines[1:]
    ids = [line.split('\t')[0] for line in lines]

    # for folder in dirs:
    #     if folder not in ids:
    #         shutil.rmtree(os.path.join(output_dir, folder))
    #         print('removed', folder)

    def remove_if_needed(folder):
        if folder not in ids:
            shutil.rmtree(os.path.join(output_dir, folder))
            print('removed', folder)

    with futures.ThreadPoolExecutor() as pool:
        tqdm(pool.map(remove_if_needed, dirs), len(dirs))


def remove_lines_manifest(source_folder):
    fs = list(map(lambda x: os.path.abspath(os.path.join(source_folder, x)), os.listdir(source_folder)))
    files = [f for f in fs if os.path.isfile(f)]
    manifest_file = [f for f in files if 'manifest' in f.lower()]
    assert len(manifest_file) == 1, 'multiple manifest files found'
    manifest_file = manifest_file[0]

    dirs = [f for f in os.listdir(source_folder) if
            not os.path.isfile(os.path.join(source_folder, f)) and not os.path.join(source_folder, f).startswith('gdc')]

    # Opens manifest and retrieves the ids to be retrieved
    with open(manifest_file, 'r') as f:
        lines = f.read().splitlines()
    headers = lines[0]
    lines = lines[1:]
    ids = [line.split('\t')[0] for line in lines]

    res = [headers]
    for line, id_slide in zip(lines, ids):
        if id_slide in dirs:
            res.append(line)

    assert len(res) == 1 + len(dirs)

    res = '\n'.join(res)
    print(res)
    open(manifest_file, 'w').write(res)
    print('rewrote manifest file', manifest_file)


def manifest_substract_manifest(m1, m2):
    with open(m1, 'r') as f:
        lines1 = f.read().splitlines()
    with open(m2, 'r') as f:
        lines2 = f.read().splitlines()
    headers = lines1[0]
    lines1 = lines1[1:]
    lines2 = lines2[1:]

    res = []
    for line in lines1:
        if line not in lines2:
            res.append(line)

    assert len(res) + len(lines2) == len(lines1)
    open(m1, 'w').write('\n'.join([headers] + res))
    print('rewrote', m1)


if __name__ == '__main__':
    # source_folder = '/home/marvinlerousseau/Documents/thesis/data/minimalist2/'
    # remove_lines_manifest(source_folder)

    m1 = '/home/marvinlerousseau/Documents/thesis/data/minimalist3/gdc_manifest_20181011_073015.txt'
    m2 = '/home/marvinlerousseau/Documents/thesis/data/minimalist3/minimalist2_gdc_manifest_20181011_073015.txt'
    manifest_substract_manifest(m1, m2)

