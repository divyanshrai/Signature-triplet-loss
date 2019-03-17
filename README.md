# Signature-triplet-loss
A one shot learning solution using a VGG19 pretrained network. A triplet loss function is used for training a last few layers. The VGG19 networks acts as a feature extractor and gives us numpy array of 512 numbers. We then compare the features and can differentiate the original signature from the forgeries

Datasets used for training 
1. A sample set of signatures available at - https://www.kaggle.com/divyanshrai/handwritten-signatures
2. ICDAR 2009 Signature Verification Competition data at - http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2009_Signature_Verification_Competition_(SigComp2009)

To test out the training code, fork the kaggle notebook - https://www.kaggle.com/divyanshrai/training-using-greyscale

Final trained weights available at - https://www.kaggle.com/divyanshrai/weights-for-signature-verification

Since keras doesn't offer a triplet loss function we define our own - 

    def triplet_loss(y_true, y_pred):
      alpha = 0.5
      anchor, positive, negative =y_pred[0,0:512], y_pred[0,512:1024], y_pred[0,1024:1536]
      positive_distance = K.mean(K.square(anchor - positive),axis=-1)
      negative_distance = K.mean(K.square(anchor - negative),axis=-1)
      return K.mean(K.maximum(0.0, positive_distance - negative_distance + alpha))

### Some important preprocessing of images

1. Using OTSU thresholding ot remove any traces of background inteference

2. Cropping and padding the larger, msotly empty images to just include the signature
