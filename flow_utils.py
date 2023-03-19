from mmflow.apis import init_model, inference_model
import numpy as np
import math

class FlowUtils:
    def __init__(self, config_file, checkpoint_file) -> None:
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file

        self.model = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        
        
    def calculate_flow(self, image1, image2):
        output = inference_model(self.model, image1, image2)
        return output


    def calculate_normalized_flow(self, image1, image2):
        output = inference_model(self.model, image1, image2)

        binary_image = np.zeros(output.shape[0:2])
        for i in range(len(output)):
            for j in range(len(output[0])):
                binary_image[i][j] = math.sqrt(output[i][j][0] ** 2 + output[i][j][1] ** 2)

        normalized_matrix = (binary_image - np.min(binary_image)) / (np.max(binary_image) - np.min(binary_image))
        binary_img = normalized_matrix * 255

        return binary_img, output

    def calculate_object_mask_from_flow(self, normalized_flow):
        pixels_with_moving_object = np.where(normalized_flow > 125)

        object_mask = np.zeros(normalized_flow.shape[0:2])
        object_mask[pixels_with_moving_object] = 255

        return object_mask
