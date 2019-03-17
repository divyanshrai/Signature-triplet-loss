# Signature-triplet-loss
A one shot learning solution using a VGG19 pretrained network. A triplet loss function is used for training a last few layers. The VGG19 networks acts as a feature extractor and gives us numpy array of 512 numbers. We then compare the features and can differentiate the original signature from the forgeries

Datasets used for training 
1. A sample set of signatures available at - https://www.kaggle.com/divyanshrai/handwritten-signatures
2. ICDAR 2009 Signature Verification Competition data at - http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2009_Signature_Verification_Competition_(SigComp2009)

