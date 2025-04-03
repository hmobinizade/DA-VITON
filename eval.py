from torchvision.transforms import transforms
import eval_models as models
from PIL import Image, ImageDraw
import os.path as osp
import lpips
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import json
import os

def evaluate_lpips_distance(image1_path, image2_path, loss_fn):
    # Load the images
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")

    # # Create the LPIPS model
    # loss_fn = lpips.LPIPS(net='alex')  # Choose a network architecture ('alex', 'vgg', 'squeeze')
    # distance_fn = lpips  # Choose the distance metric ('dist', 'emd', 'dist_unreduced')

    # Prepare the images for evaluation
    transform = transforms.Resize((768, 1024))
    image1 = transform(image1)
    image2 = transform(image2)

    # Convert the images to tensors
    image1_tensor = transforms.ToTensor()(image1).unsqueeze(0)
    image2_tensor = transforms.ToTensor()(image2).unsqueeze(0)

    # Calculate the LPIPS distance between the images
    distance = loss_fn(image1_tensor, image2_tensor).item()

    return distance


# def lpips1(name):
#     # person image
#     im_pil_big = Image.open(osp.join("data\\test\\image", name["im_name"]))
#     im_pil = transforms.Resize(768, interpolation=2)(im_pil_big)
#     im = transform(im_pil)
#
#     # target image
#     im_target_pil_big = Image.open(
#         osp.join("Output_paired", name['im_name'].split('.')[0] + '_' + name['c_name'].split('.')[0] + '.png'))
#     im_target_pil = transforms.Resize(768, interpolation=2)(im_target_pil_big)
#     im_target = transform(im_target_pil)
#
#     avg_distance = model.forward(T2(im), T2(im_target))
#
#     # avg_distance = avg_distance / 500
#     print(f"LPIPS: {avg_distance}")


def evaluate_pixel_to_pixel(image1_path, image2_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    transform = transforms.Resize((768, 1024))
    image1 = transform(image1)
    image2 = transform(image2)

    # Convert the images to RGB mode (if they are not already)
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    # Get the pixel data for both images
    pixels1 = image1.load()
    pixels2 = image2.load()

    # Check if the images have the same dimensions
    if image1.size != image2.size:
        print("Warning: Images have different dimensions.")

    # Calculate the pixel-wise distance
    distance = 0
    for i in range(image1.size[0]):  # Width
        for j in range(image1.size[1]):  # Height
            r1, g1, b1 = pixels1[i, j]
            r2, g2, b2 = pixels2[i, j]
            distance += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

    # Normalize the distance
    total_pixels = image1.size[0] * image1.size[1]
    normalized_distance = distance / (total_pixels * 3)  # Divided by 3 for RGB channels

    return normalized_distance


def mean_squared_error(image1, image2):
    # Load the two images
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    # Resize the images to have the same dimensions if necessary
    image1 = cv2.resize(image1, (768, 1024))
    image2 = cv2.resize(image2, (768, 1024))

    # Convert the images to grayscale for MSE calculation
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Convert the images to arrays
    array_image1 = np.array(gray_image1)
    array_image2 = np.array(gray_image2)

    # Calculate the squared differences and mean
    squared_diff = (array_image1 - array_image2) ** 2
    mse = np.mean(squared_diff)

    return mse


def ssim1(image1, image2):
    # Load the two images
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    # Convert the images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize the images to have the same dimensions if necessary
    gray_image1 = cv2.resize(gray_image1, (768, 1024))
    gray_image2 = cv2.resize(gray_image2, (768, 1024))

    # Calculate the SSIM
    ssim_score = ssim(gray_image1, gray_image2)
    return ssim_score

def Average(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average

if __name__ == "__main__":

    target = "condition_first_depth_cloth_front_multiheadatt_8_150p_1024"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)
    T2 = transforms.Compose([transforms.Resize((128, 128))])

    # load data list
    im_names = []
    c_names = []
    names = []
    with open("C:/hd-vton-dataset/test_pairs.txt", 'r') as f:
        for line in f.readlines():
            im_name, c_name = line.strip().split()
            names.append({
                'im_name': im_name,
                'c_name': im_name
            })

    mesures = {}

    pixel_to_pixel_distance_list = []
    lpips_distance_list = []
    mse_distance_list = []
    ssim_distance_list = []

    loss_fn = lpips.LPIPS(net='alex')

    for name in names:
        pic1 = osp.join("C:\\hd-vton-dataset\\test\\image", name["im_name"])
        pic2 = osp.join(target, name['im_name'].split('.')[0] + '_' + name['c_name'].split('.')[0] + '.png')

        pixel_to_pixel_distance = evaluate_pixel_to_pixel(pic1, pic2)
        lpips_distance = evaluate_lpips_distance(pic1, pic2, loss_fn)
        mse_distance = mean_squared_error(pic1, pic2)
        ssim_distance = ssim1(pic1, pic2)

        print('-----------------------------------------------------')
        print(pic2)
        print(f'evaluate_pixel_to_pixel: {pixel_to_pixel_distance}')
        print(f'evaluate_lpips_distance: {lpips_distance}')
        # print(f'lpips1: {lpips1(name)}')
        print(f'Mean Squared Error (MSE): {mse_distance}')
        print(f'Structural Similarity Index (SSIM): {ssim_distance}')
        print('-----------------------------------------------------')


        pixel_to_pixel_distance_list.append(pixel_to_pixel_distance)
        lpips_distance_list.append(lpips_distance)
        mse_distance_list.append(mse_distance)
        ssim_distance_list.append(ssim_distance)

        print(len(lpips_distance_list),'-',Average(lpips_distance_list))

        # mesures[pic1] = {
        #     'pic1': pic1,
        #     'pic2': pic2,
        #     'pixel_to_pixel': pixel_to_pixel_distance,
        #     'lpips': lpips_distance,
        #     'mse': mse_distance,
        #     'ssim': ssim_distance,
        # }
        #
        # print('-------------------------------')

    #
    # # Serializing json
    # json_object = json.dumps(mesures, indent=4)
    #
    # # Writing to sample.json
    # with open("my_paired.json", "w") as outfile:
    #     outfile.write(json_object)

    f = open(os.path.join(target, 'my_evaluate.txt'), 'a')
    f.write(f"SSIM : {Average(ssim_distance_list)} / MSE : {Average(mse_distance_list)} / LPIPS : {Average(lpips_distance_list)}\n")
    f.write(f"pixel_to_pixel_distance : {Average(pixel_to_pixel_distance_list)}")

    f.close()
