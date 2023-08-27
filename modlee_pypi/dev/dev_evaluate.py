
# Define your evaluation function
def evaluate_image_class(model, data_loader, device):

    import torch

    model.eval()  # Set the model to evaluation mode

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            num_correct += (predictions == labels).sum().item()
            num_samples += labels.size(0)

    accuracy = num_correct / num_samples
    return accuracy

# evaluate_text = inspect.getsource(evaluate)

