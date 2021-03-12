from model import *
from data import *
from tensorflow.keras.callbacks import EarlyStopping

#Specifies the GPU to use for CUDA
#CUDA DevKit must be installed with Cuddn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Due to the small amount of data, to provide more data for the network
# Data augmentation takes place by manipulating the input into multiple images with
# different augmentations
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='constant')

#myGene = trainGenerator(2,'data/crackconcrete/train','image','label',data_gen_args,save_to_dir = "data/crackconcrete/train/aug")
#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
imgs_train,imgs_mask_train = loadTrainNpy()
#Running the geneTrainNpy method takes a while due to going through larger amounts of augmented data

model = crf_unet()
#model = simple_unet()
early_stopping = EarlyStopping(monitor='loss', patience=6, min_delta=0.0001, mode='min')
model_checkpoint = ModelCheckpoint('unet_cracks.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=200,epochs=20,callbacks=[model_checkpoint])
model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=50, verbose=1,validation_split=0.4, shuffle=True, callbacks=[model_checkpoint])

testGene = testGenerator("data/crackForest/test")
model.load_weights("unet_cracks.hdf5")
results = model.predict_generator(testGene,118,verbose=1)
saveResult("data/crackForest/test",results)