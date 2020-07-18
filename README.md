# PointCloudModelNet
This is meant to be an interface with the online location of ModelNet data and a dataset which handles the fetching of batch info at runtime, among other things. Included as well is a (slow) rendering module for the point cloud data and its surfaced trianglulation through matplotlib's 3d offering. Currently, ModelNet40 is only offered with training data, and some of the off extension loading is still being worked out. The extension parser was written somewhat hastily and at 3 in the morning, so perhaps this is worth returning to. At any rate, .off format files, containing point cloud data, that are processed are able to be rendered, and also passed through pytorch. This is meant as a component in demonstrating the efficacy of permutation invariance as a consideration in training and inference via deepsets. 
