def load_data(ipath='./input/'):
    df_norm = pd.read_csv('./data_norm.csv') # load normalzied labels
    df = pd.read_csv('./data.csv', index_col=0) # load original labels
#     df = standardize(df)
#     df = normalize(df)
    
    def preprocess(arr, size):
        h, w = arr.shape
        s = size//2 if size%2==0 else (size+1)//2
        arr = arr[(h//2-s):(h//2+s), (w//2-s):(w//2+s)]
#         arr = np.stack((arr,)*3, axis=-1)  # to 3 channel
        return arr
    
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
        img3d = np.zeros([1,232,224,224])
        img3d_draft = np.zeros([depth,224,224])
        
        for i, ds in enumerate(slices):
            data = ds.pixel_array
            img2d = preprocess(data, 224)
#             img2d = data if data.shape[0]==H else resize(data, (224, 224), anti_aliasing=True)
            img3d_draft[i,:,:] = img2d
            
        if depth != 232:
            img3d_draft = resize_data(img3d_draft, 232, 224, 224)
        img3d[0,:,:,:] = img3d_draft
        images.append(img3d)
#         print("#", end='')
    
    df_norm['image'] = images; print(" done!")
    
    return df_norm
