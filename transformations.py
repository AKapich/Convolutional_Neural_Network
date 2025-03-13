from torchvision import transforms

def simple_transform() -> transforms:
    return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
    
def pretrained_transform() -> transforms:
    """
    Transformations to use in pretrained vgg16 and googlenet.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
