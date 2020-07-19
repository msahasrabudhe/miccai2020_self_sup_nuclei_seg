import tile_processing as tp
import os
import sys
import numpy as np

root        = './'
SLIDE_SIZE  = 224
OUT_ROOT    = '../monuseg_tiles_%dx%d/' %(SLIDE_SIZE, SLIDE_SIZE)

def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()

def run(n_imgs=None):
    directories = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    files       = []
    for d in directories:
        files_  = [f for f in os.listdir(os.path.join(root, d)) if f.endswith('.svs')]
        for f in files_:
            files.append(os.path.join(root, d, f))


    n_imgs      = len(files)

    for _id in range(n_imgs):
        s_name      = files[_id]
        # Get the name of the tile file. 
        out_name    = os.path.basename(s_name)
        # Get the name of tile without the extension. 
        out_name    = os.path.splitext(out_name)[0]
   
        savepath    = os.path.join(OUT_ROOT, out_name)
        if os.path.exists(savepath):
            print('%s already exists. No need to extract tiles again.' %(savepath))
            continue
        else:
            print('%s does not exist. Proceeding with extraction.' %(savepath))


#        if os.path.exists(os.path.join(savepath, 'slide_files/10/0_0.png')):
#            print('Skipping %s.' %(savepath))
#            continue
#        else:
#            print('0_0.png not found in %s.' %(os.path.join(savepath, 'slide_files/10/')))
#            continue

        ret     = tp.slide2tiles(s_name, OUT_ROOT, SLIDE_SIZE, savepath)
    
        write_flush('Finished extracting tiles for %3d/%3d patients.' %(_id, n_imgs))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        n_imgs = int(sys.argv[1])
    else:
        n_imgs = None
   
    print('Tile size = %d. Writing results to %s.' %(SLIDE_SIZE, OUT_ROOT))
    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)
    run(n_imgs=n_imgs)
