# Define the dataset augmentation transformations
def transform():

  import torchvision.transforms as transforms

  t = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  return t

# transform_text = inspect.getsource(transform)#transform_text_converter.get_code_text_for_transform(transform)
