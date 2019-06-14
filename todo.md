My model is just learning the mean occurrence prob of each class!
This is because it cannot extract useful information from the input data.
And that is because I dont separate the storms good enough.



preprocessing:
	strat1: static cuts
		all clustering algorithms have their drawbacks. 
			Just cut the full image into squares. 
			Analyse each square at each day for presence of storms. 
			only save
	strat2: dbscan

radarData: 
	use geopandas/raster-io for reading
	use dbscan for clustering: https://scikit-learn.org/stable/modules/clustering.html#dbscan

