# thresholded
def load_data(ipath='./cropped/'):
    df_norm = pd.read_csv('./data_norm.csv') # load normalzied labels
    df = pd.read_csv('./data.csv', index_col=0) # load original labels
    images = []
    print("in progress... ", end='')
    for fname in df.index:
        img_dir = os.listdir(ipath + fname)
        files = []
        for img in img_dir:
            files.append(dcmread(ipath + fname + '/' + img))
        
        slices = []
        skipcount = 0
        for f in files:
            if hasattr(f, 'SliceLocation'):
                slices.append(f)
            else:
                skipcount += 1
        slices = sorted(slices, key=lambda s: s.SliceLocation)
        
        depth = len(slices)
        img3d = np.zeros((1,232,224,224))
        img3d_draft = np.zeros((depth,224,224))
        
        for i, ds in enumerate(slices):
            data = ds.pixel_array
            img3d_draft[i,:,:] = data
            
        if depth != 232:
            img3d_draft = resize_data(img3d_draft,232,224,224)
        img3d[:,:,:,:] = img3d_draft
        images.append(img3d)
    df_norm['image'] = images; print(" done!")
    return df_norm
