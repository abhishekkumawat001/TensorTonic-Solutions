import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    scales = np.asarray(scales, dtype = float)
    aspect_ratios = np.asarray(aspect_ratios, dtype = float)
    
    stride = image_size / feature_size
    
    anchors = []
    
    for i in range(feature_size):
        c_y = (i + 0.5) * stride
        for j in range(feature_size):
            c_x = (j + 0.5) * stride

            for scale in scales:
                for ar in aspect_ratios:
                     w = scale * ar **0.5
                     h = scale * (1/ar**0.5)
                     anchors.append([c_x - w/2,c_y - h/2, c_x + w/2, c_y + h/2])
                    
    return anchors