import lora as l


#U-Net
#l.train_lora(
#    base_model_id="runwayml/stable-diffusion-v1-5",
#    data_folder="datasets/ds-test",
#    train_unet=True,
#    train_text_encoder=False,
#    epochs=1,
#    batch_size=1
#)

#text encoder
l.train_lora(
    base_model_id="runwayml/stable-diffusion-v1-5",
    data_folder="my_data_15",
    train_unet=False,
    train_text_encoder=True,
    epochs=1,
    batch_size=1
)
#
##both
#l.train_lora(
#    base_model_id="runwayml/stable-diffusion-v1-5",
#    data_folder="my_data_15",
#    train_unet=True,
#    train_text_encoder=True,
#    epochs=2
#)
#
##sdxl, U-Net
#l.train_lora(
#    base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
#    data_folder="my_data_xl",
#    train_unet=True,
#    train_text_encoder=False,
#    resolution=1024,
#    epochs=1,
#    batch_size=1
#)