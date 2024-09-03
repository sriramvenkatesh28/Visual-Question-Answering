import torch


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
    att_qa_answer = att_qa[:, answer].sum(axis=1)  # Get attention for answer
    ax.bar(np.arange(len(index2word)), att_qa_answer, width=0.5, color='gray')
    ax.set_xticks(np.arange(len(index2word)))
    ax.set_xticklabels([index2word[idx].strip("<pad>") for idx in question.cpu().numpy()])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Question words')
    ax.set_ylabel('Attention to answer')
    plt.show()

    # Plot attention map for image
    fig, ax = plt.subplots()
    att_img_answer = att_img[answer]  # Get attention for answer
    att_img_answer = (att_img_answer - att_img_answer.min()) / (att_img_answer.max() - att_img_answer.min())  # Normalize attention
    ax.imshow(img.squeeze().cpu().permute(1, 2, 0))
    ax.imshow(att_img_answer, cmap='jet', alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Image')
    plt.show()

    # Print example
    print(f'Example {i+1}:')
    print(f'Question: {index2word[question.cpu().numpy()].strip("<pad>")}?')
    print(f'Ground truth answer: {index2answer[answer.cpu().numpy()]}')
    print(f'Predicted answer: {index2answer[pred]}')





