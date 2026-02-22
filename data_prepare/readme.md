# Step 1: Data Prepare
    1.dataset_build.py
        -The dataset is constructed by editing distances, sample labels, and contains pairs of positive and negative example samples, 
        -Data equalization is achieved by limiting the number of times a single sample is sampled.

    2. 2.feature_extractor_TVA.py
        -Get feature of t-v-a modality, pretrained encoders are provided the same time.
        -Input the dataset_scp, a scp file whose lines are the paths of npy files, each npy file stands for one pair of enrollment and query samples. 
        -For each npy, a dict in stored, with keys like:
            1. enrollment(anc) wav path
            2. query(com) wav path
            3. enrollment(anc) lip path
            4. query(com) lip path
            5. text of the enrollment
            6. text of the query
            7. label: 0/1
            8. ....


# Step 2: Model Training and Testing.
    ./baseline/PLCL_AncTVA_ComVA_Speech_VMaskKLD_MMAlign3KernelGaussianMSE0.5_4ProjTwoLayer_Lr0.01_HalfEvery2epoch_OneVProj
    -The program is based on 4 tesla V100 32GB GPU training, with the first ten epoches of checkpoint provided together with the program.
    -Test scripts under the same project
