from PIL import Image, ImageOps

# path = 'results/GANno_pretrain_test/'
path = 'results/GANfaces_test/'
save_path = 'results/faces_pretrain.png'
nb_images = 5
size = 256

new_image = Image.new('RGB',(nb_images*size, 3*size), (250,250,250))

for i in range(nb_images):
    id = [2, 3, 4, 8, 13][i]
    im = Image.open(f'{path}test_{id:04d}.png')
    im_real = Image.open(f'{path}test_{id:04d}_real.png')
    im_gray = ImageOps.grayscale(im_real)

    new_image.paste(im_real,(i*size, 0))
    new_image.paste(im_gray,(i*size, size))
    new_image.paste(im,(i*size, 2*size))

new_image = new_image.save(save_path)