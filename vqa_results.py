import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import VqaDataset
from data_loader import get_loader
from models import VqaModel
import matplotlib.pyplot as plt
import numpy as np


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set paths to data and model
data_path = './COCO-2015/val2014/'
model_path = './models/best_model.pt'

# Set batch size
batch_size = 64

# Load dataset
val_transforms = transforms.Compose([
    transforms.Resize(448),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
print(VqaDataset(data_path, val_transforms))
val_dataset = './COCO-2015/val2014/'
print(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model
model = VqaModel().to(device)
model.load_state_dict(torch.load(model_path))

# Set model to evaluation mode
model.eval()

# Set up index to word and answer mapping
index2word = val_dataset.index2word
index2answer = val_dataset.index2answer

# Set up random indices for selecting examples
example_indices = torch.randint(0, len(val_dataset), (30,))


# Generate examples
for i in range(30):
    # Get example
    example_idx = example_indices[i]
    img, question, answer = val_dataset[example_idx]
    img = img.to(device)
    question = question.to(device)

    # Get predicted answer and attention maps
    logits, att_maps = model(img.unsqueeze(0), question.unsqueeze(0))
    pred = torch.argmax(logits).item()
    att_qa = att_maps['qa'].detach().cpu().squeeze(0).numpy()
    att_img = att_maps['img'].detach().cpu().squeeze(0).numpy()

    # Plot attention map for question
    fig, ax = plt.subplots()
    ax.imshow(att_qa.sum(axis=0), cmap='gray')
    ax.set_xticks(np.arange(len(index2word)))
    ax.set_xticklabels([index2word[idx].strip("<pad>") for idx in question.cpu().numpy()])
    ax.set_yticks([])
    plt.show()

    # Plot attention map for image
    fig, ax = plt.subplots()
    ax.imshow(att_img.sum(axis=0), cmap='gray')
    ax.imshow(img.squeeze().cpu().permute(1, 2, 0), alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    # Print example
    print(f'Example {i+1}:')
    print(f'Question: {index2word[question.cpu().numpy()].strip("<pad>")}?')
    print(f'Ground truth answer: {index2answer[answer.cpu().numpy()]}')
    print(f'Predicted answer: {index2answer[pred]}')
